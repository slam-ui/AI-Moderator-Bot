import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from llama_cpp import Llama, LlamaGrammar
import asyncio


BOT_TOKEN = "" # <-- НЕ ЗАБУДЬТЕ ВСТАВИТЬ СЮДА ВАШ ТОКЕН
MODEL_PATH = "Phi-3-mini-4k-instruct-q4.gguf"
N_GPU_LAYERS = -1  # -1 означает "выгрузить все слои на GPU"


SYSTEM_PROMPT = """
[ROLE]
Ты — AI-модератор для корпоративного чата. Твоя единственная задача — вынести вердикт 'нарушение' или 'норма' на основе правил.

[PRIMARY GOAL]
Твоя главная цель — обеспечивать бесперебойное рабочее общение. Удалять следует **ТОЛЬКО** явные нарушения. Если есть хоть малейшее сомнение, сообщение следует считать нормой. **Конструктивные предложения и короткие рабочие подтверждения ('понял', 'спасибо', 'готово') — это НОРМА.**

[TASK]
Проанализируй [USER MESSAGE] на основе [RULES]. Сначала проведи внутренний анализ по шагам (reasoning), а затем на новой строке вынеси окончательный вердикт. Твой единственный видимый ответ должен быть 'Verdict: нарушение' или 'Verdict: норма'.

[RULES]
---
{rules}
---

[EXAMPLES OF ANALYSIS WITH REASONING]
---
Сообщение: 'Я посмотрел новый макет. В целом хорошо, но мне кажется, кнопку 'Отправить' лучше сделать более заметной. Можем обсудить.'
Reasoning: Сообщение содержит конструктивное предложение по улучшению рабочего продукта. Намерения оскорбить нет.
Verdict: норма

Сообщение: 'Отлично, спасибо!'
Reasoning: Сообщение является коротким подтверждением или выражением благодарности в рамках рабочего диалога. Это не оффтоп, а часть коммуникации.
Verdict: норма

Сообщение: 'Иван, твой дизайн ужасен. Ты вообще старался?'
Reasoning: Сообщение содержит прямую негативную оценку ('ужасен') и переход на личность. Это неконструктивная критика.
Verdict: нарушение

Сообщение: 'Поздравляю всех с успешным завершением квартала! Мы — лучшая команда!'
Reasoning: Сообщение является поздравлением и не относится к оперативным рабочим задачам. Это оффтоп.
Verdict: нарушение
---

[USER MESSAGE]
---
{message}
---

[YOUR RESPONSE]
"""
NOTIFICATION_MESSAGE = "Сообщение от @{username} было удалено модератором AI из-за нарушения правил чата. Пожалуйста, ознакомьтесь с закрепленным сообщением."


logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


try:
    grammar_text = r'''
    root ::= "Verdict: " ("нарушение" | "норма")
    '''
    grammar = LlamaGrammar.from_string(grammar_text)
    logger.info("Грамматика для AI успешно создана.")
except Exception as e:
    logger.error(f"Не удалось создать грамматику: {e}")
    grammar = None

chat_rules = {}

logger.info("Загрузка модели AI-модератора...")
try:
    llm = Llama(model_path=MODEL_PATH, n_ctx=4096, n_gpu_layers=N_GPU_LAYERS, verbose=False, n_threads=8)
    logger.info("Модель AI-модератора успешно загружена!")
except Exception as e:
    logger.error(f"Ошибка при загрузке модели: {e}")
    exit()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Бот-модератор c AI-грамматикой запущен. Используйте /loadrules.")


async def load_rules_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.message.chat_id;
    user_id = update.message.from_user.id
    try:
        admins = await context.bot.get_chat_administrators(chat_id)
        is_admin = any(admin.user.id == user_id for admin in admins)
        if not is_admin:
            await update.message.reply_text("Эта команда доступна только для администраторов чата."); return
    except Exception as e:
        await update.message.reply_text(f"Ошибка проверки прав администратора: {e}"); return

    chat = await context.bot.get_chat(chat_id)
    pinned_message = chat.pinned_message
    rules_text = (pinned_message.text or pinned_message.caption) if pinned_message else None
    if rules_text:
        chat_rules[chat_id] = rules_text;
        logger.info(f"Правила для чата {chat_id} загружены.")
        await update.message.reply_text("Правила чата успешно загружены из закрепленного сообщения. Начинаю модерацию.")
    else:
        await update.message.reply_text("Не удалось найти закрепленное сообщение с текстом правил. Пожалуйста, закрепите сообщение с правилами и повторите команду.")


def generate_decision(prompt, grammar_obj):
    output = llm(prompt, max_tokens=20, grammar=grammar_obj, echo=False)
    return output['choices'][0]['text'].strip()


async def get_moderation_decision(prompt: str, grammar_obj) -> str:
    loop = asyncio.get_event_loop()
    decision = await loop.run_in_executor(None, generate_decision, prompt, grammar_obj)
    return decision


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    if chat_id not in chat_rules: return
    if grammar is None: logger.error("Грамматика не загружена, модерация отключена."); return

    user_message = update.message.text or update.message.caption
    if not user_message: return

    user = update.message.from_user
    logger.info(f"Получено сообщение от {user.first_name} для AI-проверки: '{user_message}'")

    current_rules = chat_rules[chat_id]
    full_prompt = SYSTEM_PROMPT.format(rules=current_rules, message=user_message)

    ai_decision_text = await get_moderation_decision(full_prompt, grammar)
    logger.info(f"Вердикт AI (по грамматике) для сообщения от {user.first_name}: '{ai_decision_text}'")

    if 'нарушение' in ai_decision_text.lower():
        logger.warning(f"AI решил удалить сообщение от {user.first_name}.")
        try:
            await update.message.delete()
            notification = NOTIFICATION_MESSAGE.format(username=user.username or user.first_name)
            await context.bot.send_message(chat_id=chat_id, text=notification)
        except Exception as e:
            logger.error(f"Не удалось удалить сообщение: {e}")
    else:
        logger.info("Сообщение одобрено AI.")


def main() -> None:
    application = Application.builder().token(BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("loadrules", load_rules_command))
    application.add_handler(MessageHandler(filters.TEXT | filters.CAPTION & ~filters.COMMAND, handle_message))
    application.run_polling()


if __name__ == "__main__":
    main()
import os
import base64
from flask import Flask, render_template, request, jsonify
from groq import Groq
from pypdf import PdfReader

app = Flask(__name__)

SYSTEM_INSTRUCTION = """أنت مساعد ذكاء اصطناعي متكامل وشامل (يشبه Gemini) ومفتوح المعرفة، تجيب على أي سؤال أو طلب أو كود في أي مجال بطلاقة ووضوح تام دون رفض أو تقييد.
بالإضافة إلى ذلك، عندما يطلب منك المستخدم التعلم أو التدريب في هندسة البرمجيات، تطوير الويب الشامل (Full Stack Python & Django)، الخوارزميات وهياكل البيانات، قواعد البيانات (SQL)، واجهات المستخدم (UI)، نظام لينكس، أو الأمن السيبراني، فإنك تتحول إلى مدرب ومعلم خبير يلتزم بالتالي:
1. اللغة: تحدث بنفس لغة رسالة المستخدم (العربية أو الإنجليزية).
2. التدريس من الصفر والتطبيق على VS Code: اشرح الأساسيات والأوامر من الصفر بأمثلة واضحة، ووجّه الطالب خطوة بخطوة لكيفية كتابة الكود وتشغيله في VS Code.
3. شجرة المهارات واكتشاف الموهبة: ارسم مسارات متشعبة وحلل أداء الطالب لتكتشف المسار الذي يبدع فيه.
4. تحويل الأفكار لمشاريع: حوّل أي فكرة مشروع إلى دورة تدريبية يبنيها الطالب يدوياً ويفهم طريقة عملها وتعديلها.
5. التفكير المنطقي و GitHub: اشرح المنطق قبل الكود، واطلب مشاريع عملية لـ GitHub وافحصها بدقة لكشف الأخطاء والثغرات.
6. المساعدة المفتوحة: إذا طلب المستخدم الكود كاملاً وطريقة كتابته، أعطه الكود الصحيح كاملاً واشرح كل سطر فيه بالتفصيل."""

def get_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key and os.path.exists("api_key.txt"):
        with open("api_key.txt", "r", encoding="utf-8") as f:
            api_key = f.read().strip()
    if api_key:
        return Groq(api_key=api_key)
    return None

def read_pdf(filepath):
    reader = PdfReader(filepath)
    return "\n".join([page.extract_text() or "" for page in reader.pages])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    client = get_client()
    if not client:
        return jsonify({'reply': '⚠️ مفتاح GROQ_API_KEY غير مضاف في إعدادات السيرفر.'})

    user_message = request.form.get('message', '').strip()
    file = request.files.get('file')
    image_base64 = None
    mime_type = "image/jpeg"

    if file and file.filename:
        os.makedirs('/tmp/uploads', exist_ok=True)
        filepath = os.path.join('/tmp/uploads', file.filename)
        file.save(filepath)
        ext = file.filename.lower()

        if ext.endswith(('.png', '.jpg', '.jpeg', '.webp')):
            with open(filepath, 'rb') as f:
                image_base64 = base64.b64encode(f.read()).decode('utf-8')
            if ext.endswith('.png'):
                mime_type = 'image/png'
            elif ext.endswith('.webp'):
                mime_type = 'image/webp'
            if not user_message:
                user_message = "اشرح محتوى هذه الصورة بالتفصيل، وإذا كانت تحتوي على كود أو خطأ فاشرحه وطريقة حله."
        elif ext.endswith('.pdf'):
            pdf_text = read_pdf(filepath)
            user_message += f"\n\n[محتوى ملف PDF ({file.filename})]:\n{pdf_text[:25000]}"
        else:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                file_text = f.read()
            user_message += f"\n\n[محتوى الملف ({file.filename})]:\n{file_text[:25000]}"

    if not user_message and not image_base64:
        return jsonify({'reply': 'يرجى كتابة رسالة أو إرفاق ملف أو صورة.'})

    try:
        if image_base64:
            model_name = "qwen/qwen3.8-27b"
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{SYSTEM_INSTRUCTION}\n\nطلب المستخدم: {user_message}"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}
                        }
                    ]
                }
            ]
        else:
            model_name = "openai/gpt-oss-120b"
            messages = [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": user_message}
            ]

        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model_name,
            temperature=0.6,
        )
        return jsonify({'reply': chat_completion.choices[0].message.content})
    except Exception as e:
        return jsonify({'reply': f'حدث خطأ أثناء الاتصال بـ Groq: {str(e)}'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)



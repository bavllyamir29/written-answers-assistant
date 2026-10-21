import os
from typing import Any
from urllib.parse import urljoin

import requests
import streamlit as st


DEFAULT_BACKEND_URL = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT = (3.05, 180)

st.set_page_config(
    page_title="مساعد تصحيح الإجابات",
    page_icon="📝",
    layout="centered",
)

st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        direction: rtl;
        text-align: right;
    }
    [data-testid="stAppViewContainer"] {
        direction: rtl;
    }
    [data-testid="stSidebar"] {
        direction: rtl;
        text-align: right;
    }
    div[data-testid="stFormSubmitButton"] button {
        width: 100%;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _grade_endpoint(base_url: str) -> str:
    return urljoin(f"{base_url.rstrip('/')}/", "grade")


def send_to_backend(payload: dict[str, Any], base_url: str) -> dict[str, Any]:
    """Send the existing backend's GradeRequest and return its JSON response."""
    response = requests.post(
        _grade_endpoint(base_url),
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    try:
        body = response.json()
    except requests.exceptions.JSONDecodeError as exc:
        raise ValueError("أعاد الخادم استجابة غير صالحة بصيغة JSON.") from exc

    if not response.ok:
        detail = body.get("detail", body) if isinstance(body, dict) else body
        raise RuntimeError(f"HTTP {response.status_code}: {detail}")
    if not isinstance(body, dict):
        raise ValueError("استجابة الخادم ليست كائن JSON كما هو متوقع.")
    return body


def _display_result(result: dict[str, Any]) -> None:
    st.subheader("نتيجة التقييم")
    score = result.get("score")
    max_score = result.get("max_score")
    if score is not None and max_score is not None:
        st.metric("الدرجة", f"{score} / {max_score}")

    score_columns = st.columns(3)
    score_columns[0].metric(
        "التقارب الدلالي", f"{result.get('semantic_similarity', 0):.2f}"
    )
    score_columns[1].metric(
        "مطابقة الكلمات", f"{result.get('keyword_match_ratio', 0):.2f}"
    )
    score_columns[2].metric(
        "اكتمال الإجابة", f"{result.get('completeness_ratio', 0):.2f}"
    )

    feedback = result.get("feedback_ar")
    if feedback:
        st.markdown("**التعليق**")
        st.info(feedback)

    matched = result.get("matched_keywords") or []
    missing = result.get("missing_keywords") or []
    if matched:
        st.markdown("**مصطلحات مطابقة:** " + "، ".join(map(str, matched)))
    if missing:
        st.markdown("**مصطلحات لم تظهر:** " + "، ".join(map(str, missing)))

    flags = result.get("review_flags") or []
    if flags:
        st.markdown("**ملاحظات للمراجعة**")
        for flag in flags:
            st.warning(str(flag))

    with st.expander("تفاصيل النتيجة"):
        st.json(result)


st.title("📝 مساعد تصحيح الإجابات")
st.caption("أدخل السؤال والإجابة النموذجية، ثم أرسل إجابة الطالب للتقييم.")

with st.sidebar:
    st.header("إعداد الاتصال")
    backend_url = st.text_input(
        "عنوان الخادم الخلفي",
        value=DEFAULT_BACKEND_URL,
        help="عنوان FastAPI الأساسي، مثل http://127.0.0.1:8000",
    ).strip()
    st.caption("سيتم إرسال الإجابة إلى المسار POST /grade.")
    st.caption("اترك معرّف الطالب فارغًا إذا لم تكن هناك حاجة إليه.")

with st.form("grading_form", clear_on_submit=False):
    question = st.text_area(
        "السؤال الحالي",
        placeholder="اكتب سؤال المقال هنا...",
        height=110,
    )
    model_answer = st.text_area(
        "الإجابة النموذجية للمعلم",
        placeholder="اكتب الإجابة النموذجية التي سيقارن بها النظام...",
        height=150,
    )
    student_answer = st.text_area(
        "إجابة الطالب",
        placeholder="اكتب إجابة الطالب هنا...",
        height=180,
    )

    with st.expander("إعدادات إضافية"):
        id_columns = st.columns(2)
        question_id = id_columns[0].text_input("معرّف السؤال (اختياري)")
        student_id = id_columns[1].text_input("معرّف الطالب (اختياري)")
        max_score = st.number_input(
            "الدرجة الكلية",
            min_value=0.01,
            max_value=1000.0,
            value=5.0,
            step=0.5,
        )
        keyword_text = st.text_area(
            "كلمات أو عبارات مطلوبة (اختياري، عبارة واحدة في كل سطر)",
            placeholder="البناء الضوئي\nضوء الشمس",
            height=100,
        )
        extract_keywords = st.checkbox(
            "استخراج كلمات مقترحة تلقائيًا إذا لم أكتب كلمات مطلوبة",
            value=False,
        )

    submitted = st.form_submit_button("إرسال الإجابة للتقييم")

if submitted:
    if not backend_url:
        st.error("أدخل عنوان الخادم الخلفي.")
    elif not question.strip():
        st.error("أدخل نص السؤال.")
    elif not model_answer.strip():
        st.error("أدخل الإجابة النموذجية.")
    elif not student_answer.strip():
        st.error("أدخل إجابة الطالب.")
    else:
        payload: dict[str, Any] = {
            "question": question.strip(),
            "model_answer": model_answer.strip(),
            "student_answer": student_answer.strip(),
            "max_score": max_score,
            "extract_keywords": extract_keywords,
            "allow_fuzzy_matching": True,
            "weights": {
                "semantic": 0.7,
                "keywords": 0.2,
                "completeness": 0.1,
            },
        }
        keywords = [line.strip() for line in keyword_text.splitlines() if line.strip()]
        if keywords:
            payload["must_have_keywords"] = keywords
        if question_id.strip():
            payload["question_id"] = question_id.strip()
        if student_id.strip():
            payload["student_id"] = student_id.strip()

        try:
            with st.spinner("جارٍ إرسال الإجابة وانتظار نتيجة التقييم..."):
                result = send_to_backend(payload, backend_url)
            st.session_state["last_grade_result"] = result
        except requests.exceptions.Timeout:
            st.error("انتهت مهلة الاتصال بالخادم. تحقق من تشغيله ثم حاول مرة أخرى.")
        except requests.exceptions.ConnectionError:
            st.error("تعذر الاتصال بالخادم. تحقق من العنوان ومن تشغيل FastAPI.")
        except (requests.exceptions.RequestException, RuntimeError, ValueError) as exc:
            st.error(f"تعذر إكمال طلب التقييم: {exc}")

if "last_grade_result" in st.session_state:
    _display_result(st.session_state["last_grade_result"])

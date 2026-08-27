# src/engine/chatbot.py
# Orchestrator chinh: cau hoi -> phan loai y dinh (diem so / danh sach lop /
# thoi khoa bieu / ...) -> trich xuat bo loc (ten/lop/nam hoc/hoc ky) ->
# tra cuu (co ap dung gioi han theo vai tro dang nhap) -> LLM dien giai ket qua.

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Generator, List, Optional, Set, Tuple, Any

from config import (
    LOG_FORMAT, LOG_LEVEL, GRADES_DIR, DEFAULT_LLM_PROVIDER,
    USE_SUPABASE, USE_AUTH, SUPABASE_URL, SUPABASE_KEY, SUPABASE_ANON_KEY, SUPABASE_SUBJECT_NAME,
)
from src.grades.grade_store import GradeStore, GradeRecord, normalize_name
from src.memory.memory import MemoryManager, DEFAULT_SESSION_ID
from src.llm.prompt_templates import (
    build_grade_prompt, build_no_match_prompt,
    build_roster_prompt, build_no_roster_params_prompt,
    build_timetable_prompt, build_no_timetable_params_prompt,
    build_attendance_prompt, build_no_attendance_match_prompt,
    build_exam_schedule_prompt, build_no_exam_params_prompt,
    build_notifications_prompt,
    build_activities_prompt,
    build_student_info_prompt,
    build_summary_prompt,
    build_class_stats_prompt,
    build_teacher_prompt,
    build_feature_unavailable_prompt,
    build_permission_denied_prompt,
)
from src.grades.analytics import summarize_student, class_stats
from src.llm.llm_chain import call_llm, call_llm_streaming
from src.llm.response_builder import build_final_response, grade_citation_lines, format_for_display, FinalResponse

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


_STUDENT_CODE_RE = re.compile(r"\bHS\d{3,}\b", re.IGNORECASE)

# Cac cum bao hieu nguoi dung muon xem diem CUA TAT CA CAC NAM HOC (khong gioi han nam hien tai).
_ALL_YEARS_KEYWORDS = [
    "qua cÃ¡c nÄƒm", "qua tá»«ng nÄƒm", "táº¥t cáº£ cÃ¡c nÄƒm", "toÃ n bá»™ cÃ¡c nÄƒm", "táº¥t cáº£ nÄƒm há»c",
    "má»i nÄƒm", "cÃ¡c nÄƒm há»c", "háº¿t cÃ¡c nÄƒm", "qua cÃ¡c nÄƒm há»c", "toÃ n bá»™ Ä‘iá»ƒm qua",
    "tá»« trÆ°á»›c Ä‘áº¿n nay", "tá»« trÆ°á»›c tá»›i nay", "lá»‹ch sá»­ Ä‘iá»ƒm",
]

def wants_all_years(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in _ALL_YEARS_KEYWORDS)

def analyze_query_llm(
    question: str,
    history: Optional[List[dict]] = None,
    prev_context: Optional[dict] = None,
) -> dict:
    """Su dung LLM de phan loai y dinh va trich xuat filter cung 1 luc.

    prev_context (optional): ngu canh hoi thoai tu luot truoc, dang
    {"intent": "...", "filters": {"name_query": ..., ...}}.  Duoc chen vao
    prompt de LLM quyet dinh ke thua hay ghi de tung filter â€” giup xu ly cau
    hoi noi tiep ("con hoc ky 2?", "diem Toan cua ban ay?") tot hon."""
    from src.llm.llm_chain import get_llm
    from langchain_core.messages import SystemMessage, HumanMessage
    
    context_str = ""
    if history:
        recent = [h["content"] for h in history[-4:] if h["role"] == "user"]
        if recent:
            context_str = "\nCÃ¡c cÃ¢u há»i trÆ°á»›c Ä‘Ã³: " + " | ".join(recent)

    # --- Khoi ngu canh luot truoc ---
    prev_context_str = ""
    if prev_context and prev_context.get("filters"):
        pf = prev_context["filters"]
        parts = []
        label_map = {
            "name_query": "TÃªn há»c sinh",
            "class_name": "Lá»›p",
            "school_year": "NÄƒm há»c",
            "semester": "Há»c ká»³",
            "subject": "MÃ´n há»c",
        }
        for key, label in label_map.items():
            val = pf.get(key)
            if val:
                parts.append(f"  - {label}: {val}")
        if parts:
            prev_intent = prev_context.get("intent", "")
            prev_context_str = (
                f"\n\nNGá»® Cáº¢NH Tá»ª CÃ‚U Há»ŽI TRÆ¯á»šC (intent trÆ°á»›c: {prev_intent}):\n"
                + "\n".join(parts)
                + "\n\nQUY Táº®C Káº¾ THá»ªA NGá»® Cáº¢NH:\n"
                "- Náº¿u cÃ¢u há»i hiá»‡n táº¡i lÃ  cÃ¢u ná»‘i tiáº¿p (thiáº¿u tÃªn/lá»›p/nÄƒm/mÃ´n...), "
                "hÃ£y Káº¾ THá»ªA cÃ¡c giÃ¡ trá»‹ tá»« ngá»¯ cáº£nh trÆ°á»›c mÃ  cÃ¢u há»i hiá»‡n táº¡i khÃ´ng Ä‘á» cáº­p.\n"
                "- Chá»‰ GHI ÄÃˆ khi ngÆ°á»i dÃ¹ng nÃªu rÃµ giÃ¡ trá»‹ má»›i (vd: nÃ³i tÃªn khÃ¡c, lá»›p khÃ¡c).\n"
                "- Náº¿u cÃ¢u há»i hoÃ n toÃ n khÃ¡c chá»§ Ä‘á» (intent khÃ¡c háº³n), KHÃ”NG káº¿ thá»«a "
                "tÃªn há»c sinh / mÃ£ HS tá»« ngá»¯ cáº£nh trÆ°á»›c.\n"
            )

    prompt = f"""Báº¡n lÃ  má»™t chuyÃªn gia phÃ¢n tÃ­ch ngá»¯ nghÄ©a. Nhiá»‡m vá»¥ cá»§a báº¡n lÃ  phÃ¢n tÃ­ch cÃ¢u há»i cá»§a ngÆ°á»i dÃ¹ng vÃ  tráº£ vá» Má»˜T chuá»—i JSON duy nháº¥t chá»©a Ã½ Ä‘á»‹nh vÃ  cÃ¡c thÃ´ng tin cáº§n thiáº¿t.
    
CHá»ˆ TRáº¢ Vá»€ JSON Há»¢P Lá»†, KHÃ”NG BAO Gá»’M Báº¤T Ká»² VÄ‚N Báº¢N NÃ€O KHÃC (KHÃ”NG DÃ™NG ```json ... ``` MARKDOWN).

Ã Ä‘á»‹nh (intent) pháº£i lÃ  Má»˜T TRONG CÃC Tá»ª KHÃ“A sau:
- timetable: Há»i vá» thá»i khÃ³a biá»ƒu, lá»‹ch há»c, lá»‹ch há»c thÃªm.
- exam: Há»i vá» lá»‹ch thi, lá»‹ch kiá»ƒm tra.
- attendance: Há»i vá» Ä‘iá»ƒm danh, Ä‘i há»c hay váº¯ng há»c, nghá»‰ há»c.
- notification: Há»i vá» thÃ´ng bÃ¡o.
- activity: Há»i vá» hoáº¡t Ä‘á»™ng ngoáº¡i khÃ³a, sá»± kiá»‡n.
- teacher: Há»i vá» giÃ¡o viÃªn (chá»§ nhiá»‡m, bá»™ mÃ´n, ai dáº¡y).
- class_stats: Thá»‘ng kÃª Ä‘iá»ƒm cá»§a lá»›p, xáº¿p háº¡ng cá»§a cáº£ lá»›p.
- summary: Tá»•ng káº¿t, xáº¿p loáº¡i há»c lá»±c cá»§a cÃ¡ nhÃ¢n, káº¿t quáº£ há»c táº­p.
- student_info: ThÃ´ng tin há»“ sÆ¡ cÃ¡ nhÃ¢n, phá»¥ huynh, liÃªn há»‡.
- roster: Danh sÃ¡ch lá»›p, sÄ© sá»‘, cÃ¡c báº¡n trong lá»›p.
- grade: Tra cá»©u Ä‘iá»ƒm sá»‘ mÃ´n há»c (Máº¶C Äá»ŠNH náº¿u khÃ´ng khá»›p cÃ¡c loáº¡i trÃªn).

Äá»‹nh dáº¡ng JSON yÃªu cáº§u:
{{
  "intent": "tá»«_khÃ³a_intent",
  "filters": {{
    "student_name": "TÃªn há»c sinh náº¿u cÃ³ (vÃ­ dá»¥: Nguyá»…n VÄƒn An), hoáº·c mÃ£ HS náº¿u cÃ³",
    "class_name": "TÃªn lá»›p náº¿u cÃ³ (vÃ­ dá»¥: 6A, 7B). Viáº¿t hoa chá»¯ cÃ¡i.",
    "school_year": "NÄƒm há»c náº¿u cÃ³ (vÃ­ dá»¥: 2023-2024, 2024-2025).",
    "semester": "Há»c ká»³ náº¿u cÃ³, CHá»ˆ GHI 'I' hoáº·c 'II'.",
    "subject": "TÃªn mÃ´n há»c náº¿u cÃ³ (vÃ­ dá»¥: ToÃ¡n, Ngá»¯ VÄƒn, Tiáº¿ng Anh, Khoa há»c tá»± nhiÃªn, v.v.)"
  }}
}}
Náº¿u khÃ´ng tÃ¬m tháº¥y thÃ´ng tin cho má»™t filter nÃ o Ä‘Ã³, hÃ£y Ä‘á»ƒ giÃ¡ trá»‹ lÃ  null.
Äá»‘i vá»›i mÃ´n há»c, cá»‘ gáº¯ng chuyá»ƒn tÃªn mÃ´n viáº¿t táº¯t hoáº·c khÃ´ng dáº¥u (vd: khtn, ly, gdcd) vá» tÃªn chuáº©n (Khoa há»c tá»± nhiÃªn, GiÃ¡o dá»¥c cÃ´ng dÃ¢n...).
TÃªn há»c sinh: HÃ£y loáº¡i bá» cÃ¡c Ä‘áº¡i tá»« xÆ°ng hÃ´, chá»‰ láº¥y tÃªn riÃªng.
{prev_context_str}
{context_str}
CÃ¢u há»i hiá»‡n táº¡i: {question}"""

    try:
        llm = get_llm(DEFAULT_LLM_PROVIDER)
        response = llm.invoke([
            SystemMessage(content="You are a JSON data extractor."),
            HumanMessage(content=prompt)
        ])
        content = response.content
        if isinstance(content, list):
            content = " ".join([str(c.get("text", "")) if isinstance(c, dict) else str(c) for c in content])
        
        content = str(content).strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        data = json.loads(content)
        
        if "filters" not in data:
            data["filters"] = {}
        filters = data["filters"]
        
        return {
            "intent": data.get("intent", "grade"),
            "filters": {
                "name_query": filters.get("student_name"),
                "class_name": filters.get("class_name"),
                "school_year": filters.get("school_year"),
                "semester": filters.get("semester"),
                "subject": filters.get("subject")
            }
        }
    except Exception as e:
        logger.error(f"Lá»—i khi dÃ¹ng LLM analyze_query: {e}")
        return {
            "intent": "grade",
            "filters": {"name_query": None, "class_name": None, "school_year": None, "semester": None, "subject": None}
        }




@dataclass
class _LookupResult:
    prompt: str
    citations: List[str] = field(default_factory=list)
    has_data: bool = False
    # notice_only: cau tra loi mang tinh THONG BAO (tu choi quyen, thieu tham
    # so, tinh nang chua kha dung) â€” KHONG phai "khong tim thay du lieu", nen
    # khong hien canh bao "vui long cung cap them thong tin".
    notice_only: bool = False


# Gioi han so ban ghi dua vao prompt de tranh phinh to context khi hoi chung
# chung (1 hoc sinh x 16 mon x 6 hoc ky ~ 96 ban ghi; nhieu ten fuzzy co the hon).
_MAX_PROMPT_RECORDS = 250


def _limit_records(records: List[GradeRecord]) -> List[GradeRecord]:
    if len(records) <= _MAX_PROMPT_RECORDS:
        return records
    logger.info("Cat bot ban ghi dua vao prompt: %d -> %d", len(records), _MAX_PROMPT_RECORDS)
    return records[:_MAX_PROMPT_RECORDS]


# ---------------------------------------------------------------------------
# ChatbotEngine â€” tra cuu diem / danh sach lop / thoi khoa bieu / ...
#
# LUU Y QUAN TRONG VE DA NGUOI DUNG: instance nay la 1 singleton dung chung
# cho toan bo server (Streamlit @st.cache_resource), moi nguoi dung dang nhap
# deu goi chung 1 ChatbotEngine. Vi vay KHONG duoc luu thong tin nguoi dung
# hien tai (vai tro, student_id...) vao thuoc tinh cua self â€” phai truyen
# session_user theo tung loi goi chat()/chat_streaming() de tranh lo du lieu
# giua cac nguoi dung dang dang nhap dong thoi.
# ---------------------------------------------------------------------------

class ChatbotEngine:
    def __init__(self):
        self.memory = MemoryManager()
        self.school_info = None
        self.auth = None

        if USE_SUPABASE:
            from src.grades.supabase_store import SupabaseGradeStore
            from src.grades.school_info import SchoolInfoStore
            self.store = SupabaseGradeStore(SUPABASE_URL, SUPABASE_KEY, SUPABASE_SUBJECT_NAME)
            self.school_info = SchoolInfoStore(SUPABASE_URL, SUPABASE_KEY)
            logger.info("Nguon du lieu diem: Supabase")
        else:
            self.store = GradeStore(GRADES_DIR)
            logger.info("Nguon du lieu diem: Excel (%s)", GRADES_DIR)

        if USE_AUTH:
            from src.auth.auth_service import AuthService
            self.auth = AuthService(SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_KEY)
            logger.info("Dang nhap/phan quyen: BAT (Supabase Auth)")
        else:
            logger.info("Dang nhap/phan quyen: TAT (khong co SUPABASE_ANON_KEY)")

        self._is_ready = False
        logger.info("ChatbotEngine da khoi tao (chua load du lieu diem)")

    def initialize(self) -> bool:
        self.store.load()
        self._is_ready = self.store.is_ready()
        if not self._is_ready:
            logger.warning("Chua co du lieu diem trong %s", GRADES_DIR)
        return self._is_ready

    def reload_index(self) -> bool:
        return self.initialize()

    def is_ready(self) -> bool:
        return self._is_ready

    def get_index_stats(self) -> dict:
        if not self._is_ready:
            return {"status": "Chua khoi tao"}
        return self.store.stats()

    # -- bo chon vai tro (demo) â€” tao SessionUser khong qua dang nhap -------

    def list_students_for_picker(self) -> List[dict]:
        """Danh sach (full_name, student_code) duy nhat de chon o UI demo.
        Lay tu du lieu diem da nap (khong goi them DB)."""
        seen = {}
        for r in self.store.records:
            if r.student_id and r.student_id not in seen:
                seen[r.student_id] = r.name
        return sorted(
            ({"student_code": code, "full_name": name} for code, name in seen.items()),
            key=lambda x: x["full_name"],
        )

    def make_demo_user(self, role_name: str, student_code: Optional[str] = None):
        """Tao SessionUser demo cho bo chon vai tro (khong xac thuc that)."""
        from src.auth.auth_service import SessionUser, ROLE_ADMIN, ROLE_TEACHER, ROLE_STUDENT

        if role_name == ROLE_STUDENT:
            full_name = student_code or ""
            student_id = None
            if student_code and self.school_info is not None:
                stu = self.school_info.get_student_by_code(student_code)
                if stu:
                    student_id = stu.get("student_id")
                    full_name = stu.get("full_name") or full_name
            if not full_name:
                # Excel mode / khong co school_info: lay ten tu du lieu diem
                for r in self.store.records:
                    if r.student_id == student_code:
                        full_name = r.name
                        break
            return SessionUser(
                user_id=-3, email="", full_name=full_name or "Há»c sinh",
                role_name=ROLE_STUDENT, student_id=student_id, student_code=student_code,
            )

        if role_name == ROLE_TEACHER:
            return SessionUser(user_id=-2, email="", full_name="GiÃ¡o viÃªn (demo)", role_name=ROLE_TEACHER)
        return SessionUser(user_id=-1, email="", full_name="Quáº£n trá»‹ viÃªn (demo)", role_name=ROLE_ADMIN)

    # -- tra cuu diem (mac dinh) ------------------------------------------

    def _current_school_year(self) -> Optional[str]:
        """Nam hoc HIEN TAI tu Supabase (is_current=TRUE).
        Neu khong co (che do Excel) thi lay nam moi nhat co trong du lieu diem."""
        if self.school_info is not None:
            try:
                ans = self.school_info.get_current_school_year()
                if ans:
                    return ans
            except Exception as e:
                logger.warning("Khong lay duoc nam hoc hien tai: %s", e)
        years = self.store.list_school_years()
        return years[-1] if years else None

    def _resolve_year_filter(self, question: str, explicit_year: Optional[str]) -> Optional[str]:
        """Xac dinh nam hoc can loc khi tra cuu diem:
        - Neu cau hoi neu ro nam (vd 2024-2025) -> dung nam do.
        - Neu hoi "qua cac nam / tat ca cac nam" -> None (khong loc, lay het).
        - Mac dinh (hoi chung chung) -> nam hoc HIEN TAI."""
        if explicit_year:
            return explicit_year
        if wants_all_years(question):
            return None
        return self._current_school_year()

    def _resolve_records(
        self, question: str, filters: dict, forced_student_code: Optional[str] = None,
    ) -> Tuple[List[GradeRecord], List[str]]:
        """Tra ve (danh sach ban ghi khop, danh sach ten goi y neu khong khop).

        forced_student_code: neu duoc truyen (hoc sinh dang dang nhap), bo qua
        hoan toan viec tim ten trong cau hoi va CHI loc theo ma hoc sinh nay â€”
        ngan hoc sinh xem duoc diem cua nguoi khac bang cach go ten khac.

        Neu cau hoi co ten mon cu cu the -> chi tra diem mon do; neu hoi chung
        chung -> tra diem TAT CA cac mon."""
        name_query = filters.get("name_query")
        subject = filters.get("subject")

        # Nam hoc: mac dinh nam hien tai; neu neu ro nam -> nam do; neu hoi
        # "qua cac nam" -> tat ca (None).
        year_filter = self._resolve_year_filter(question, filters["school_year"])

        # Xac dinh danh sach MA hoc sinh can lay diem (viec khop ten dua tren
        # chi muc ten da nap; ma/ten it thay doi). Sau do lay DIEM MOI NHAT truc
        # tiep tu nguon (real-time voi Supabase) qua fetch_for_codes().
        if forced_student_code:
            codes = [forced_student_code]
            suggestions: List[str] = []
        else:
            if not name_query:
                return [], []
            matched_names = self.store.find_matching_names(name_query)
            if not matched_names:
                return [], []
            suggestions = matched_names
            nameset = set(matched_names)
            codes = sorted({r.student_id for r in self.store.records if r.name in nameset and r.student_id})
            if not codes:
                return [], matched_names

        fresh = self.store.fetch_for_codes(codes)
        records = [
            r for r in fresh
            if (not filters["class_name"] or r.class_name.upper() == filters["class_name"].upper())
            and (not year_filter or r.school_year == year_filter)
            and (not filters["semester"] or r.semester == filters["semester"])
            and (not subject or r.subject == subject)
        ]
        # Sap xep theo (hoc sinh, mon, nam, hoc ky) de HK I luon lien truoc HK II
        # cua cung mon -> LLM trinh bay dung, khong dao HK.
        records.sort(key=lambda r: (r.name, r.subject, r.school_year, r.semester))
        return _limit_records(records), suggestions

    # -- danh sach lop / thoi khoa bieu -----------------------------------

    def _resolve_roster(self, question: str, filters: dict) -> _LookupResult:
        class_name, school_year = filters.get("class_name"), filters.get("school_year")

        if self.school_info is None:
            return _LookupResult(build_feature_unavailable_prompt(question, "danh sÃ¡ch lá»›p"))

        missing = []
        if not class_name:
            missing.append("tÃªn lá»›p")
        if not school_year:
            missing.append("nÄƒm há»c")
        if missing:
            return _LookupResult(build_no_roster_params_prompt(question, missing), notice_only=True)

        roster = self.school_info.get_class_roster(class_name, school_year)
        prompt = build_roster_prompt(question, class_name, school_year, roster)
        citations = [f"Danh sÃ¡ch lá»›p {class_name} - NÄƒm há»c {school_year}"] if roster else []
        return _LookupResult(prompt, citations, bool(roster))

    def _resolve_timetable(self, question: str, filters: dict, session_user=None) -> _LookupResult:
        if self.school_info is None:
            return _LookupResult(build_feature_unavailable_prompt(question, "thá»i khÃ³a biá»ƒu"))

        class_name, school_year, semester = filters.get("class_name"), filters.get("school_year"), filters.get("semester")

        # HOC SINH: mac dinh LUON xem TKB CUA CHINH MINH trong ky hien tai,
        # ke ca khi hoi chung chung (vd chi go "tkb") â€” khong can noi "cua toi".
        is_student = session_user is not None and getattr(session_user, "is_student", False)
        if is_student and session_user.student_id is not None:
            cur = self.school_info.get_current_term(date.today().isoformat())
            year_name = school_year or (cur["year_name"] if cur else None)
            if year_name:
                resolved_class = self.school_info.get_student_class(session_user.student_id, year_name)
                if resolved_class:
                    class_name, school_year = resolved_class, year_name

        missing = []
        if not class_name:
            missing.append("tÃªn lá»›p")
        if not school_year:
            missing.append("nÄƒm há»c")
        if missing:
            return _LookupResult(build_no_timetable_params_prompt(question, missing), notice_only=True)

        # Xac dinh hoc ky: uu tien "hoc ky 1/2" neu neu ro; nguoc lai dung ky
        # hien tai (real-time) khi cung nam hoc.
        if semester:
            term_order = 2 if semester == "II" else 1
            term_label = f"Há»c ká»³ {semester}"
        elif cur and school_year == cur.get("year_name"):
            term_order = cur.get("term_order")
            term_label = f"Há»c ká»³ {'II' if term_order == 2 else 'I'} (hiá»‡n táº¡i)"
        else:
            term_order = None
            term_label = None

        # TKB 1 hoc ky la CO DINH -> lay 1 tuan dai dien (tuan hien tai theo ngay
        # thuc) va loc dung hoc ky de tranh lan du lieu giua 2 ky trong cung tuan.
        week_start = self.school_info.pick_representative_week(
            class_name, school_year, date.today().isoformat()
        )
        rows = self.school_info.get_timetable(
            class_name, school_year, week_start=week_start, term_order=term_order
        )

        prompt = build_timetable_prompt(question, class_name, school_year, term_label, rows)
        cite = f"Thá»i khÃ³a biá»ƒu lá»›p {class_name} - NÄƒm há»c {school_year}"
        if term_label:
            cite += f" - {term_label}"
        citations = [cite] if rows else []
        return _LookupResult(prompt, citations, bool(rows))

    def _resolve_attendance(
        self, question: str, filters: dict, forced_student_id: Optional[int] = None, forced_full_name: Optional[str] = None,
    ) -> _LookupResult:
        if self.school_info is None:
            return _LookupResult(build_feature_unavailable_prompt(question, "Ä‘iá»ƒm danh"))

        if forced_student_id is not None:
            student_ids = [forced_student_id]
            display_name = forced_full_name or ""
        else:
            name_query = filters.get("name_query")
            if not name_query:
                return _LookupResult(build_no_attendance_match_prompt(question, []))

            matched_names = self.store.find_matching_names(name_query)
            if not matched_names:
                return _LookupResult(build_no_attendance_match_prompt(question, []))

            students = self.school_info.find_student_ids_by_names(matched_names)
            if not students:
                return _LookupResult(build_no_attendance_match_prompt(question, matched_names))

            student_ids = [s["student_id"] for s in students]
            display_name = students[0]["full_name"] if students else ""

        records = self.school_info.get_attendance(student_ids)
        prompt = build_attendance_prompt(question, records)
        citations = [f"Äiá»ƒm danh cá»§a {display_name}"] if records else []
        return _LookupResult(prompt, citations, bool(records))

    def _resolve_exam_schedule(self, question: str, filters: dict, session_user=None) -> _LookupResult:
        if self.school_info is None:
            return _LookupResult(build_feature_unavailable_prompt(question, "lá»‹ch thi"))

        class_name, school_year, semester = filters.get("class_name"), filters.get("school_year"), filters.get("semester")

        # Hoc ky HIEN TAI theo ngay thuc (co fallback ve ky gan nhat khi nghi he)
        cur = self.school_info.get_current_term(date.today().isoformat())

        # HOC SINH: mac dinh LUON xem lich thi CUA CHINH MINH trong ky hien tai,
        # ke ca khi hoi chung chung (vd chi go "lich thi") â€” khong can noi "cua
        # toi", va khong xem duoc lich thi lop khac. Tu suy ra lop tu ho so.
        is_student = session_user is not None and getattr(session_user, "is_student", False)
        if is_student and session_user.student_id is not None:
            year_name = school_year or (cur["year_name"] if cur else None)
            if year_name:
                resolved_class = self.school_info.get_student_class(session_user.student_id, year_name)
                if resolved_class:
                    class_name, school_year = resolved_class, year_name

        missing = []
        if not class_name:
            missing.append("tÃªn lá»›p")
        if not school_year:
            missing.append("nÄƒm há»c")
        if missing:
            return _LookupResult(build_no_exam_params_prompt(question, missing), notice_only=True)

        # Xac dinh hoc ky: uu tien "hoc ky 1/2" neu neu ro; nguoc lai dung ky
        # hien tai (real-time) khi cung nam hoc.
        if semester:
            term_order = 2 if semester == "II" else 1
            term_label = f"Há»c ká»³ {semester}"
        elif cur and school_year == cur.get("year_name"):
            term_order = cur.get("term_order")
            suffix = "hiá»‡n táº¡i" if cur.get("is_current") else "gáº§n nháº¥t"
            term_label = f"Há»c ká»³ {'II' if term_order == 2 else 'I'} ({suffix})"
        else:
            term_order = None
            term_label = None

        rows = self.school_info.get_exam_schedule(class_name, school_year, term_order=term_order)
        # Ky hien tai chua co lich thi va nguoi dung khong chi ro hoc ky -> hien
        # lich thi hien co cua lop (moi ky) de van tra ket qua huu ich.
        if not rows and term_order and not semester:
            rows = self.school_info.get_exam_schedule(class_name, school_year, term_order=None)
            if rows:
                term_label = "lá»‹ch thi hiá»‡n cÃ³"

        prompt = build_exam_schedule_prompt(question, class_name, school_year, term_label, rows)
        cite = f"Lá»‹ch thi lá»›p {class_name} - NÄƒm há»c {school_year}"
        if term_label:
            cite += f" - {term_label}"
        citations = [cite] if rows else []
        return _LookupResult(prompt, citations, bool(rows))

    def _resolve_student_info(self, question: str, filters: dict) -> _LookupResult:
        if self.school_info is None:
            return _LookupResult(build_feature_unavailable_prompt(question, "thÃ´ng tin há»c sinh"))

        # Uu tien tra cuu theo ma hoc sinh neu cau hoi co (vd HS00457)
        codes = [m.group(0).upper() for m in _STUDENT_CODE_RE.finditer(question)]
        if codes:
            profiles = self.school_info.get_student_profiles(codes=codes)
            citations = [f"ThÃ´ng tin há»c sinh {p.get('student_code')}" for p in profiles]
            return _LookupResult(build_student_info_prompt(question, profiles), citations, bool(profiles))

        name_query = filters.get("name_query")
        if not name_query:
            return _LookupResult(build_student_info_prompt(question, []))

        matched_names = self.store.find_matching_names(name_query)
        if not matched_names:
            return _LookupResult(build_student_info_prompt(question, []))

        profiles = self.school_info.get_student_profiles(names=matched_names)
        citations = [f"ThÃ´ng tin há»c sinh: {p.get('full_name')}" for p in profiles]
        return _LookupResult(build_student_info_prompt(question, profiles), citations, bool(profiles))

    # -- tong ket / xep loai (Thong tu 22) --------------------------------

    def _resolve_summary(self, question: str, filters: dict, forced_student_code: Optional[str] = None) -> _LookupResult:
        name_query = filters.get("name_query")
        school_year = filters.get("school_year")
        semester = filters.get("semester")
        target = "I" if semester == "I" else "II" if semester == "II" else "year"

        # Xac dinh ma hoc sinh roi lay diem MOI NHAT truc tiep tu nguon (real-time)
        if forced_student_code:
            codes = [forced_student_code]
            display_name = None
        else:
            if not name_query:
                return _LookupResult(build_no_match_prompt(question, []))
            matched = self.store.find_matching_names(name_query)
            if not matched:
                return _LookupResult(build_no_match_prompt(question, []))
            names = set(matched)
            codes = sorted({r.student_id for r in self.store.records if r.name in names and r.student_id})
            display_name = matched[0] if matched else None
        recs = self.store.fetch_for_codes(codes)
        if not recs:
            return _LookupResult(build_no_match_prompt(question, []))

        # Chon nam hoc: uu tien nam duoc neu ro, nguoc lai lay nam moi nhat co du lieu
        if school_year:
            recs = [r for r in recs if r.school_year == school_year]
        else:
            years = sorted({r.school_year for r in recs})
            school_year = years[-1] if years else None
            recs = [r for r in recs if r.school_year == school_year]

        student_name = display_name or (recs[0].name if recs else "")
        if not recs:
            return _LookupResult(build_no_match_prompt(question, []))

        term_word = "Há»c ká»³ I" if target == "I" else "Há»c ká»³ II" if target == "II" else "cáº£ nÄƒm"
        term_label = f"{term_word} nÄƒm há»c {school_year}"
        summary = summarize_student(recs, target)
        has = bool(summary.get("numeric") or summary.get("nhanxet"))
        prompt = build_summary_prompt(question, student_name, term_label, summary)
        cite = f"Tá»•ng káº¿t {term_label} - {student_name}"
        return _LookupResult(prompt, [cite] if has else [], has)

    # -- thong ke lop (giao vien / admin) ---------------------------------

    def _resolve_class_stats(self, question: str, filters: dict) -> _LookupResult:
        class_name = filters.get("class_name")
        school_year = filters.get("school_year")
        semester = filters.get("semester")
        subject = filters.get("subject")
        target = "I" if semester == "I" else "II" if semester == "II" else "year"

        if not class_name:
            return _LookupResult(
                f"Cau hoi hoi ve thong ke lop nhung chua ro TEN LOP.\n\n"
                f"Cau hoi cua nguoi dung: {question}\n\n"
                f"Hay lich su de nghi nguoi dung cho biet ten lop (vd 6A) va nam hoc de thong ke.",
                notice_only=True,
            )

        # Danh sach ma hoc sinh cua lop (lay tu chi muc da nap â€” DS lop on dinh),
        # sau do lay diem MOI NHAT truc tiep tu nguon (real-time).
        cache_recs = [r for r in self.store.records if r.class_name.upper() == class_name.upper()]
        if school_year:
            cache_recs = [r for r in cache_recs if r.school_year == school_year]
        else:
            years = sorted({r.school_year for r in cache_recs})
            school_year = years[-1] if years else None
            cache_recs = [r for r in cache_recs if r.school_year == school_year]
        codes = sorted({r.student_id for r in cache_recs if r.student_id})

        fresh = self.store.fetch_for_codes(codes)
        recs = [r for r in fresh if not school_year or r.school_year == school_year]

        term_word = "Há»c ká»³ I" if target == "I" else "Há»c ká»³ II" if target == "II" else "cáº£ nÄƒm"
        term_label = f"{term_word}" if school_year else None
        stats = class_stats(recs, subject, target)
        has = bool(stats.get("num_students"))
        prompt = build_class_stats_prompt(question, class_name, school_year or "(chÆ°a rÃµ)", term_label, subject, stats)
        cite = f"Thá»‘ng kÃª lá»›p {class_name}"
        if school_year:
            cite += f" - {school_year}"
        if subject:
            cite += f" - {subject}"
        return _LookupResult(prompt, [cite] if has else [], has)

    # -- tra cuu giao vien ------------------------------------------------

    def _resolve_teacher(self, question: str, filters: dict, include_contact: bool = True, session_user=None) -> _LookupResult:
        if self.school_info is None:
            return _LookupResult(build_feature_unavailable_prompt(question, "tra cá»©u giÃ¡o viÃªn"))

        class_name = filters.get("class_name")
        school_year = filters.get("school_year")
        subject = filters.get("subject")
        name_query = filters.get("name_query")
        q_low = question.lower()
        wants_homeroom = any(k in q_low for k in ["chá»§ nhiá»‡m", "gvcn"])

        # HOC SINH: mac dinh LUON tra cuu giao vien (chu nhiem/bo mon) cua LOP MINH,
        # neu khong neu ro ten lop.
        is_student = session_user is not None and getattr(session_user, "is_student", False)
        if is_student and session_user.student_id is not None:
            cur = self.school_info.get_current_term(date.today().isoformat())
            year_name = school_year or (cur["year_name"] if cur else None)
            if year_name:
                resolved_class = self.school_info.get_student_class(session_user.student_id, year_name)
                if resolved_class:
                    class_name, school_year = resolved_class, year_name

        if not school_year:
            years = self.store.list_school_years()
            school_year = years[-1] if years else None

        def _teacher_line(t: dict) -> str:
            parts = [f"{t.get('full_name', '')}"]
            if t.get("teacher_code"):
                parts.append(f"(MÃ£ GV: {t['teacher_code']})")
            if t.get("subject_name"):
                parts.append(f"- mÃ´n: {t['subject_name']}")
            if t.get("title"):
                parts.append(f"- chá»©c vá»¥: {t['title']}")
            if include_contact and t.get("phone"):
                parts.append(f"- SÄT: {t['phone']}")
            return "- " + " ".join(parts)

        # 1) Theo lop
        if class_name and school_year:
            if wants_homeroom:
                t = self.school_info.get_homeroom_teacher(class_name, school_year)
                header = f"GiÃ¡o viÃªn chá»§ nhiá»‡m lá»›p {class_name} nÄƒm há»c {school_year}:"
                lines = [_teacher_line(t)] if t else []
                cite = f"GVCN lá»›p {class_name} - {school_year}"
                return _LookupResult(build_teacher_prompt(question, header, lines),
                                     [cite] if lines else [], bool(lines))

            teachers = self.school_info.get_class_teachers(class_name, school_year)
            if subject:
                teachers = [t for t in teachers if t.get("subject_name") == subject]
                header = f"GiÃ¡o viÃªn dáº¡y mÃ´n {subject} lá»›p {class_name} nÄƒm há»c {school_year}:"
            else:
                header = f"GiÃ¡o viÃªn bá»™ mÃ´n lá»›p {class_name} nÄƒm há»c {school_year}:"
            lines = [f"- {t['subject_name']}: {t['full_name']}"
                     + (f" (MÃ£ GV: {t['teacher_code']})" if t.get("teacher_code") else "")
                     for t in teachers]
            cite = f"GiÃ¡o viÃªn lá»›p {class_name} - {school_year}"
            return _LookupResult(build_teacher_prompt(question, header, lines),
                                 [cite] if lines else [], bool(lines))

        # 2) Theo ten / ma giao vien
        if name_query:
            teachers = self.school_info.find_teachers_by_name(name_query)
            if teachers:
                header = "ThÃ´ng tin giÃ¡o viÃªn:"
                lines = []
                for t in teachers:
                    lines.append(_teacher_line(t))
                    assign = self.school_info.get_teacher_assignments(t["teacher_id"], school_year)
                    for h in assign.get("homeroom", []):
                        lines.append(f"    â€¢ Chá»§ nhiá»‡m lá»›p {h['class_name']} (nÄƒm {h['year_name']})")
                    taught = assign.get("teaching", [])
                    if taught:
                        pairs = ", ".join(
                            f"{a['class_name']}"
                            + (f"/{a['subject_name']}" if a.get("subject_name") else "")
                            for a in taught
                        )
                        lines.append(f"    â€¢ Dáº¡y: {pairs}")
                cite = f"ThÃ´ng tin giÃ¡o viÃªn: {teachers[0].get('full_name')}"
                return _LookupResult(build_teacher_prompt(question, header, lines), [cite], True)

        return _LookupResult(build_teacher_prompt(question, "", []))

    def _resolve_notifications(self, question: str) -> _LookupResult:
        if self.school_info is None:
            return _LookupResult(build_feature_unavailable_prompt(question, "thÃ´ng bÃ¡o"))

        notifications = self.school_info.get_recent_notifications()
        prompt = build_notifications_prompt(question, notifications)
        citations = ["ThÃ´ng bÃ¡o nhÃ  trÆ°á»ng (gáº§n Ä‘Ã¢y nháº¥t)"] if notifications else []
        return _LookupResult(prompt, citations, bool(notifications))

    def _resolve_activities(self, question: str, filters: dict) -> _LookupResult:
        if self.school_info is None:
            return _LookupResult(build_feature_unavailable_prompt(question, "hoáº¡t Ä‘á»™ng ngoáº¡i khÃ³a"))

        school_year, semester = filters.get("school_year"), filters.get("semester")

        activities = self.school_info.get_activities(school_year, semester)
        prompt = build_activities_prompt(question, activities)
        cite = "Hoáº¡t Ä‘á»™ng ngoáº¡i khÃ³a"
        if school_year:
            cite += f" - NÄƒm há»c {school_year}"
        if semester:
            cite += f" - Há»c ká»³ {semester}"
        citations = [cite] if activities else []
        return _LookupResult(prompt, citations, bool(activities))

    # -- phan quyen theo vai tro dang nhap ----------------------------------


    def _get_tools(self, session_user, context: dict):
        from langchain_core.tools import tool
        from pydantic import Field
        
        def _apply_result(lookup_res, title="Tra cứu"):
            if lookup_res.citations:
                context["citations"].extend(lookup_res.citations)
            context["has_data"] = context["has_data"] or lookup_res.has_data
            context["tools_used"].append({"tool": title, "summary": title})
            return lookup_res.prompt

        @tool
        def tra_cuu_diem(student_name: str = Field(None, description="Tên hoặc mã học sinh"), class_name: str = Field(None, description="Lớp"), subject: str = Field(None, description="Môn học"), school_year: str = Field(None, description="Năm học"), semester: str = Field(None, description="Học kỳ I hoặc II")):
            """Tra cứu điểm số môn học của học sinh."""
            filters = {"name_query": student_name, "class_name": class_name, "school_year": school_year, "semester": semester, "subject": subject}
            forced = session_user.student_code if session_user and getattr(session_user, "is_student", False) else None
            if not filters.get("school_year") and not wants_all_years(student_name or ""):
                filters["school_year"] = self._current_school_year()
            records, suggestions = self._resolve_records("Tra cứu điểm", filters, forced_student_code=forced)
            if not records:
                return _apply_result(_LookupResult(build_no_match_prompt("Tra cứu điểm", suggestions)), "Tra cứu điểm")
            return _apply_result(_LookupResult(build_grade_prompt("Tra cứu điểm", records), grade_citation_lines(records), True), "Tra cứu điểm")

        @tool
        def xem_thoi_khoa_bieu(class_name: str = Field(None, description="Lớp"), school_year: str = Field(None, description="Năm học"), semester: str = Field(None, description="Học kỳ I hoặc II")):
            """Tra cứu thời khóa biểu, lịch học."""
            filters = {"class_name": class_name, "school_year": school_year, "semester": semester}
            return _apply_result(self._resolve_timetable("Xem thời khóa biểu", filters, session_user), "Xem thời khóa biểu")

        @tool
        def xem_lich_thi(class_name: str = Field(None, description="Lớp"), school_year: str = Field(None, description="Năm học"), semester: str = Field(None, description="Học kỳ I hoặc II")):
            """Tra cứu lịch thi, lịch kiểm tra."""
            filters = {"class_name": class_name, "school_year": school_year, "semester": semester}
            return _apply_result(self._resolve_exam_schedule("Xem lịch thi", filters, session_user), "Xem lịch thi")

        @tool
        def xem_diem_danh(student_name: str = Field(None, description="Tên hoặc mã học sinh")):
            """Tra cứu lịch sử điểm danh, nghỉ học."""
            filters = {"name_query": student_name}
            forced_id = session_user.student_id if session_user and getattr(session_user, "is_student", False) else None
            forced_name = getattr(session_user, "full_name", None) if forced_id else None
            return _apply_result(self._resolve_attendance("Xem điểm danh", filters, forced_student_id=forced_id, forced_full_name=forced_name), "Xem điểm danh")

        @tool
        def xem_thong_tin_hoc_sinh(name_or_code: str = Field(..., description="Tên hoặc mã học sinh")):
            """Tra cứu thông tin cá nhân, hồ sơ học sinh."""
            if session_user and getattr(session_user, "is_student", False):
                return _apply_result(_LookupResult(build_permission_denied_prompt("Tra cứu thông tin", "tra cứu thông tin học sinh"), notice_only=True), "Tra cứu thông tin")
            filters = {"name_query": name_or_code}
            return _apply_result(self._resolve_student_info(name_or_code, filters), "Tra cứu hồ sơ học sinh")

        @tool
        def tong_ket_hoc_tap(student_name: str = Field(None, description="Tên học sinh"), school_year: str = Field(None, description="Năm học"), semester: str = Field(None, description="Học kỳ I hoặc II")):
            """Tra cứu tổng kết, xếp loại học lực của học sinh."""
            filters = {"name_query": student_name, "school_year": school_year, "semester": semester}
            forced = session_user.student_code if session_user and getattr(session_user, "is_student", False) else None
            return _apply_result(self._resolve_summary("Tổng kết", filters, forced_student_code=forced), "Tổng kết học tập")

        @tool
        def thong_ke_lop(class_name: str = Field(..., description="Lớp"), school_year: str = Field(None, description="Năm học"), semester: str = Field(None, description="Học kỳ I hoặc II"), subject: str = Field(None, description="Môn học")):
            """Xem thống kê điểm số, xếp hạng của cả lớp."""
            if session_user and getattr(session_user, "is_student", False):
                return _apply_result(_LookupResult(build_permission_denied_prompt("Thống kê", "thống kê điểm cả lớp"), notice_only=True), "Thống kê lớp")
            filters = {"class_name": class_name, "school_year": school_year, "semester": semester, "subject": subject}
            return _apply_result(self._resolve_class_stats("Thống kê", filters), "Thống kê lớp")

        @tool
        def tra_cuu_giao_vien(name_or_code: str = Field(None, description="Tên hoặc mã GV"), class_name: str = Field(None, description="Lớp"), subject: str = Field(None, description="Môn học"), school_year: str = Field(None, description="Năm học")):
            """Tra cứu thông tin giáo viên chủ nhiệm, bộ môn."""
            filters = {"name_query": name_or_code, "class_name": class_name, "subject": subject, "school_year": school_year}
            include_contact = not (session_user is not None and getattr(session_user, "is_student", False))
            return _apply_result(self._resolve_teacher("Tra cứu GV", filters, include_contact=include_contact, session_user=session_user), "Tra cứu giáo viên")

        @tool
        def xem_thong_bao():
            """Xem các thông báo mới nhất của nhà trường."""
            return _apply_result(self._resolve_notifications("Xem thông báo"), "Xem thông báo")

        @tool
        def xem_hoat_dong(school_year: str = Field(None, description="Năm học"), semester: str = Field(None, description="Học kỳ")):
            """Xem các hoạt động ngoại khóa, sự kiện."""
            filters = {"school_year": school_year, "semester": semester}
            return _apply_result(self._resolve_activities("Xem hoạt động", filters), "Xem hoạt động ngoại khóa")

        @tool
        def danh_sach_lop(class_name: str = Field(..., description="Lớp"), school_year: str = Field(..., description="Năm học")):
            """Xem danh sách học sinh trong lớp (sĩ số)."""
            if session_user and getattr(session_user, "is_student", False):
                return _apply_result(_LookupResult(build_permission_denied_prompt("Danh sách lớp", "danh sách lớp"), notice_only=True), "Danh sách lớp")
            filters = {"class_name": class_name, "school_year": school_year}
            return _apply_result(self._resolve_roster("Danh sách lớp", filters), "Xem danh sách lớp")

        return [tra_cuu_diem, xem_thoi_khoa_bieu, xem_lich_thi, xem_diem_danh, xem_thong_tin_hoc_sinh, tong_ket_hoc_tap, thong_ke_lop, tra_cuu_giao_vien, xem_thong_bao, xem_hoat_dong, danh_sach_lop]

    @staticmethod
    def _session_id_for(session_user) -> str:
        return session_user.session_key if session_user is not None else DEFAULT_SESSION_ID

    # -- chat (sync) ----------------------------------------------------

    def chat(self, question: str, provider: str = DEFAULT_LLM_PROVIDER, session_user=None) -> FinalResponse:
        if not self._is_ready:
            raise RuntimeError("Engine chua san sang. Chua co du lieu diem trong thu muc data/diem_khtn.")

        stream_gen = self.chat_streaming(question, provider, session_user)
        full_text = ""
        citations = []
        has_data = False
        tools_used = []
        notice_only = False

        for event_type, data in stream_gen:
            if event_type == "text":
                full_text += data
            elif event_type == "done":
                citations = data.get("citations", [])
                has_data = data.get("has_data", False)
                tools_used = data.get("tools_used", [])

        # The chat_streaming already saved the history, but wait, finalize_streaming_response used to do it.
        # Wait, chat_streaming now does `self.memory.add_assistant_message` before it finishes!
        
        from src.llm.response_builder import FinalResponse, format_for_display
        return FinalResponse(
            answer_text=full_text,
            citations=citations,
            warnings=[],
            metadata={"has_data": has_data, "tools_used": tools_used}
        )

    # -- chat (streaming) -------------------------------------------------


    def chat_streaming(
        self, question: str, provider: str = DEFAULT_LLM_PROVIDER, session_user=None,
    ) -> Generator[Tuple[str, Any], None, None]:
        from src.llm.llm_chain import get_llm, build_messages
        from langchain_core.messages import AIMessage, ToolMessage
        import json

        if not self._is_ready:
            raise RuntimeError("Engine chua san sang. Chua co du lieu diem trong thu muc data/diem_khtn.")

        session_id = self._session_id_for(session_user)
        self.memory.add_user_message(question, session_id=session_id)

        context = {"citations": [], "has_data": False, "tools_used": []}
        tools = self._get_tools(session_user, context)
        
        llm = get_llm(provider)
        if hasattr(llm, "bind_tools"):
            llm_with_tools = llm.bind_tools(tools)
        else:
            llm_with_tools = llm

        messages = build_messages(question, self.memory.get_chat_history(session_id), session_user)
        tool_map = {t.name: t for t in tools}

        loop_count = 0
        final_answer = ""
        while loop_count < 5:  # Max 5 iterations to prevent infinite loops
            loop_count += 1
            tool_call_accumulated = []
            text_accumulated = ""
            is_tool_call = False

            try:
                for chunk in llm_with_tools.stream(messages):
                    if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
                        is_tool_call = True
                        tool_call_accumulated.append(chunk)
                    else:
                        content = chunk.content
                        if isinstance(content, str) and content:
                            text_accumulated += content
                            yield ("text", content)
                        elif isinstance(content, list):
                            for c in content:
                                if isinstance(c, str):
                                    text_accumulated += c
                                    yield ("text", c)
                                elif isinstance(c, dict) and "text" in c:
                                    text_accumulated += c["text"]
                                    yield ("text", c["text"])
            except Exception as e:
                logger.error(f"Lỗi streaming: {e}")
                yield ("text", f"\n[Lỗi kết nối: {e}]")
                break

            if not is_tool_call:
                final_answer = text_accumulated
                messages.append(AIMessage(content=text_accumulated))
                break

            # Reconstruct tool calls
            full_chunk = tool_call_accumulated[0]
            for c in tool_call_accumulated[1:]:
                full_chunk = full_chunk + c

            messages.append(full_chunk)

            for tc in full_chunk.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]

                yield ("tool", {"tool": tool_name, "summary": f"Đang gọi {tool_name}...", "data": {"columns": [], "rows": []}})

                if tool_name in tool_map:
                    try:
                        tool_result = tool_map[tool_name].invoke(tool_args)
                        result_str = str(tool_result)
                    except Exception as e:
                        result_str = f"Error: {e}"
                else:
                    result_str = f"Error: Tool {tool_name} not found."

                messages.append(ToolMessage(content=result_str, tool_call_id=tc["id"]))

        # Filter out unique citations
        unique_cites = list(dict.fromkeys(context["citations"]))
        yield ("done", {
            "full_answer": final_answer,
            "citations": unique_cites,
            "has_data": context["has_data"],
            "tools_used": context["tools_used"]
        })
        self.memory.add_assistant_message(message=final_answer, question=question, session_id=session_id)
    def finalize_streaming_response(
        self,
        full_text: str,
        lookup: _LookupResult,
        question: str,
        session_user=None,
    ) -> FinalResponse:
        response = build_final_response(
            llm_answer=full_text, citations=lookup.citations, has_data=lookup.has_data, question=question,
            suppress_no_data_warning=lookup.notice_only,
        )
        session_id = self._session_id_for(session_user)
        self.memory.add_assistant_message(message=full_text, question=question, session_id=session_id)
        return response

    def clear_session(self, session_user=None) -> None:
        self.memory.clear_session(self._session_id_for(session_user))

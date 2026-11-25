"""
STEP 7 - TEST YOUR CHATBOT
--------------------------
Two kinds of checks:
  A) Retrieval check (free, no API): is the expected file in the top-k results?
  B) Answer check (uses Gemini API): does the answer contain the expected fact?

Run:   python -m rag.step7_evaluate            (retrieval only)
       python -m rag.step7_evaluate --answers  (also calls Gemini)

Add more test cases as you find questions the bot gets wrong!
"""
import sys
import time

from rag.step4_retrieve import retrieve

# (question, file that should be retrieved, text that should appear in the answer)
TEST_CASES = [
    ("What is the B.Tech fee per year?", "02_fee_structure.md", "1,18,000"),
    ("Can I pay fees in installments?", "02_fee_structure.md", "15 July"),
    ("What is the late fee fine?", "02_fee_structure.md", "100"),
    ("Where does the CSE HOD sit?", "04_departments_and_hods.md", "A-301"),
    ("When can I meet Dr. Anjali Sharma?", "04_departments_and_hods.md", "Wednesday"),
    ("Is the IT HOD available on Saturday?", "04_departments_and_hods.md", "Saturday"),
    ("Who is the HOD of Mechanical?", "04_departments_and_hods.md", "Prakash Joshi"),
    ("What time does college start?", "03_college_timings.md", "9:00"),
    ("Library timings on Sunday?", "03_college_timings.md", "10:00 AM"),
    ("When are the end semester exams?", "03_college_timings.md", "December"),
    ("How much attendance is needed?", "07_exams_and_rules.md", "75%"),
    ("How do I get a bonafide certificate?", "07_exams_and_rules.md", "A-003"),
    ("Girls hostel in-time?", "06_hostel_transport_facilities.md", "8:30"),
    ("What is the highest placement package?", "06_hostel_transport_facilities.md", "24 LPA"),
    ("Eligibility for MBA?", "05_admissions.md", "CAT"),
    # Out-of-scope: the bot should say it doesn't know (no expected file)
    ("Who won the IPL last year?", None, "don't have"),
]


def run(check_answers=False):
    passed = 0
    for question, expected_file, expected_text in TEST_CASES:
        hits = retrieve(question)
        files = [h["source"] for h in hits]
        ok = expected_file is None or expected_file in files
        line = f"{'PASS' if ok else 'FAIL'} retrieval | {question}"

        if check_answers:
            from rag.step5_generate import answer
            time.sleep(4)  # free tier allows ~15 requests/min; pace calls to avoid 429
            reply, _ = answer(question)
            ans_ok = expected_text.lower() in reply.lower()
            ok = ok and ans_ok
            line += f" | answer {'PASS' if ans_ok else 'FAIL'}"
            if not ans_ok:
                line += f"\n      got: {reply[:150]!r}"
        else:
            if not ok:
                line += f"\n      top files: {files}"

        passed += ok
        print(line)
    print(f"\n{passed}/{len(TEST_CASES)} passed")


if __name__ == "__main__":
    run(check_answers="--answers" in sys.argv)

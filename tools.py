from langchain_core.tools import tool

GRADE_BANDS = [
    (85, 101, 4.0),
    (80, 85, 3.7),
    (75, 80, 3.3),
    (70, 75, 3.0),
    (65, 70, 2.7),
    (61, 65, 2.3),
    (58, 61, 2.0),
    (55, 58, 1.7),
    (50, 55, 1.0),
    (0, 50, 0.0),
]

#calculating the grade points from raw marks
@tool
def marks_to_grade_points(marks: int) -> float:
    """Use this to convert a single course's numeric marks (0-100) into PUCIT grade points.

    Call this before calculate_semester_gpa whenever the student gives raw marks
    instead of already-known grade points, once per course.
    """
    if marks < 0 or marks > 100:
        return "Error: marks must be between 0 and 100"
    for low, high, points in GRADE_BANDS:
        if low <= marks < high:
            return points
    return "Error: could not resolve marks to a grade point"


#calculating semster gpa
@tool
def calculate_semester_gpa(grade_points: list[float], credit_hours: list[float]) -> float:
    """Use this to calculate one semester's credit-weighted GPA.

    grade_points and credit_hours must be lists of equal length, one entry per
    graded course, in the same order. Do not include Math Deficiency courses
    (MD-001, MD-002); they are non-credit pass/fail and excluded from GPA.
    """
    if len(grade_points) != len(credit_hours):
        return "Error: grade_points and credit_hours must be the same length"
    if len(grade_points) == 0:
        return "Error: at least one course is required"
    total_credit_hours = sum(credit_hours)
    if total_credit_hours <= 0:
        return "Error: total credit hours must be greater than zero"
    total_quality_points = 0
    for gp, ch in zip(grade_points, credit_hours):
        total_quality_points += gp * ch
    return round(total_quality_points / total_credit_hours, 3)

#calculating cgpa by knowing this semster gpa and previous semsters cgpa
@tool
def calculate_new_cgpa(
    current_cgpa: float,
    completed_credit_hours: float,
    semester_gpa: float,
    semester_credit_hours: float,
) -> float:
    """Use this to project a student's new CGPA after completing one more semester.

    current_cgpa is the CGPA before this semester, completed_credit_hours is the
    total credit hours completed before this semester, semester_gpa is the GPA
    just earned this semester, and semester_credit_hours is this semester's
    credit hours.
    """
    if current_cgpa < 0 or current_cgpa > 4.0:
        return "Error: current_cgpa must be between 0.0 and 4.0"
    if completed_credit_hours < 0:
        return "Error: completed_credit_hours cannot be negative"
    if semester_gpa < 0 or semester_gpa > 4.0:
        return "Error: semester_gpa must be between 0.0 and 4.0"
    if semester_credit_hours <= 0:
        return "Error: semester_credit_hours must be greater than zero"
    total_credit_hours = completed_credit_hours + semester_credit_hours
    total_quality_points = (
        current_cgpa * completed_credit_hours + semester_gpa * semester_credit_hours
    )
    return round(total_quality_points / total_credit_hours, 3)


@tool
def required_gpa_for_target(
    target_cgpa: float,
    current_cgpa: float,
    completed_credit_hours: float,
    remaining_credit_hours: float,
) -> float:
    """Use this to calculate the GPA needed over the remaining credit hours to
    reach a target CGPA by graduation.

    The result can come out above 4.0, meaning the target is not reachable in
    that many remaining credit hours. Do not present a value above 4.0 as a
    real answer; instead widen remaining_credit_hours (more semesters, using
    get_remaining_credit_hours) and try again until the result is at or below
    4.0, or tell the student it is not achievable at all.
    """
    if target_cgpa < 0 or target_cgpa > 4.0:
        return "Error: target_cgpa must be between 0.0 and 4.0"
    if current_cgpa < 0 or current_cgpa > 4.0:
        return "Error: current_cgpa must be between 0.0 and 4.0"
    if completed_credit_hours < 0:
        return "Error: completed_credit_hours cannot be negative"
    if remaining_credit_hours <= 0:
        return "Error: remaining_credit_hours must be greater than zero"
    total_credit_hours = completed_credit_hours + remaining_credit_hours
    numerator = (
        target_cgpa * total_credit_hours - current_cgpa * completed_credit_hours
    )
    return round(numerator / remaining_credit_hours, 3)

COURSES = {
1: [("MS-251", "Probability & Statistics", 3.0),
("GE-160", "Applications of ICT", 3.0),
("GE-169", "Applied Physics", 3.0),
("GE-167", "Discrete Structures", 3.0),
("HQ-001", "Quran Translation - I", 0.5),
("GE-190", "Functional English", 3.0)],

2: [("CC-112", "Programming Fundamentals", 3.0),
("CC-112-L", "Programming Fundamentals Lab", 1.0),
("CC-110", "Digital Logic Design", 2.0),
("CC-110-L", "Digital Logic Design Lab", 1.0),
("MS-252", "Linear Algebra", 3.0),
("GE-191", "Expository Writing", 3.0),
("GE-163", "Islamic Studies", 2.0),
("HQ-002", "Quran Translation - II", 0.5)],

3: [("CC-211", "Object Oriented Programming", 3.0),
("CC-211-L", "Object Oriented Programming Lab", 1.0),
("CC-215", "Database Systems", 3.0),
("CC-215-L", "Database Systems Lab", 1.0),
("CC-210", "Computer Organization & Assembly Language", 3.0),
("GE-162", "Calculus & Analytical Geometry", 3.0),
("GE-192", "Introduction to Management", 2.0),
("HQ-003", "Quran Translation - III", 0.5)],

4: [("CC-213", "Data Structures", 3.0),
("CC-213-L", "Data Structures Lab", 1.0),
("CC-312", "Information Security", 3.0),
("CC-214", "Computer Networks", 3.0),
("CC-212", "Software Engineering", 3.0),
("DC-220", "Advanced Database Management Systems", 3.0),
("HQ-004", "Quran Translation - IV", 0.5)],

5: [("CC-313", "Analysis of Algorithms", 3.0),
("CC-310", "Artificial Intelligence", 3.0),
("DC-320", "Theory of Automata and Formal Languages", 3.0),
("DC-321", "Human Computer Interaction", 3.0),
("DC-322", "Computer Architecture", 3.0),
("EC-330", "Web Technologies / Elective", 3.0),
("HQ-005", "Quran Translation - V", 0.5)],

6: [("CC-311", "Operating Systems", 3.0),
("EC-333", "Mobile Application Development / Elective", 3.0),
("EC-324", "Software Construction & Development / Elective", 3.0),
("EC-335", "Machine Learning / Elective", 3.0),
("EC-334", "Game Design and Development / Elective", 3.0),
("MS-253", "Multivariable Calculus", 3.0),
("HQ-006", "Quran Translation - VI", 0.5)],

7: [("CC-411", "Final Year Project - I", 2.0),
("DC-328", "Parallel & Distributed Computing", 3.0),
("EC-345", "Computer Vision / Elective", 3.0),
("EC-425", "Software Quality Engineering / Elective", 3.0),
("MS-254", "Technical and Business Writing", 3.0),
("GE-263", "Entrepreneurship", 2.0),
("GE-262", "Professional Practices", 2.0),
("HQ-007", "Quran Translation - VII", 0.5)],

8: [("CC-412", "Final Year Project - II", 4.0),
("DC-421", "Compiler Construction", 3.0),
("UE-272", "Introduction to Marketing", 3.0),
("GE-168", "Ideology and Constitution of Pakistan", 2.0),
("GE-363", "Civics and Community Engagement", 2.0),
("HQ-008", "Quran Translation - VIII", 0.5)],
}

#get semster courses
@tool
def get_semester_courses(semester: int) -> str:
    """Use this to look up the courses and credit hours for one semester of the
    PUCIT BS(CS) scheme of studies.

    Call this when the student names a semester but has not listed its courses,
    or when you need to know a semester's total credit hours.
    """
    if semester < 1 or semester > 8:
        return "Error: semester must be between 1 and 8"
    lines = []
    for code, name, credit_hours in COURSES[semester]:
        lines.append(f"{code} - {name} ({credit_hours} ch)")
    return "\n".join(lines)

#get remaining credit hours
@tool
def get_remaining_credit_hours(current_semester: int, through_semester: int = 8) -> float:
    """Use this to total the graded credit hours from just after current_semester
    through through_semester (inclusive), for checking one horizon at a time when
    building a multi-semester plan toward a target CGPA. Defaults to semester 8
    (everything remaining) if through_semester is not given.
    """
    if current_semester < 1 or current_semester > 8:
        return "Error: current_semester must be between 1 and 8"
    if through_semester < 1 or through_semester > 8:
        return "Error: through_semester must be between 1 and 8"
    if through_semester <= current_semester:
        return "Error: through_semester must be after current_semester"
    total_credit_hours = 0.0
    for semester in range(current_semester + 1, through_semester + 1):
        for code, name, credit_hours in COURSES[semester]:
            total_credit_hours += credit_hours
    return round(total_credit_hours, 2)

#write report
@tool
def save_report(filename: str, content: str) -> str:
    """Use this to save a summary of a completed GPA or CGPA calculation to a
    text file, only after the student has asked you to save it.

    Never call this after just a clarifying question or a single grade lookup.
    """
    if not filename.strip():
        return "Error: filename cannot be empty"
    safe_name = filename if filename.endswith(".txt") else filename + ".txt"
    with open(safe_name, "w") as file:
        file.write(content)
    return f"Saved report to {safe_name}"


TOOLS = [
    marks_to_grade_points,
    calculate_semester_gpa,
    calculate_new_cgpa,
    required_gpa_for_target,
    get_semester_courses,
    get_remaining_credit_hours,
    save_report,
]
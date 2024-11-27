from dsmltf import build_tree_id3, classify

def main():
    """
    Основная функция, запускающая классификацию данных для интервью и студентов.
    Выводит результаты предсказаний на основе построенных деревьев решений.
    """
    # Дерево 1: Дерево собеседования
    interview_data = [
        ({'level': 'Senior', 'lang': 'Java', 'tweets': 'no', 'phd': 'no'}, False),
        ({'level': 'Senior', 'lang': 'Java', 'tweets': 'no', 'phd': 'yes'}, False),
        ({'level': 'Mid', 'lang': 'Python', 'tweets': 'no', 'phd': 'no'}, True),
        ({'level': 'Junior', 'lang': 'Python', 'tweets': 'no', 'phd': 'no'}, True),
        ({'level': 'Junior', 'lang': 'Python', 'tweets': 'no', 'phd': 'yes'}, False),
        ({'level': 'Senior', 'lang': 'Python', 'tweets': 'yes', 'phd': 'yes'}, True),
        ({'level': 'Senior', 'lang': 'Kotlin', 'tweets': 'no', 'phd': 'no'}, True),
        ({'level': 'Mid', 'lang': 'Swift', 'tweets': 'yes', 'phd': 'no'}, False),
        ({'level': 'Junior', 'lang': 'Rust', 'tweets': 'no', 'phd': 'no'}, True),
        ({'level': 'Mid', 'lang': 'Haskell', 'tweets': 'no', 'phd': 'yes'}, True),
        ({'level': 'Senior', 'lang': 'Elixir', 'tweets': 'yes', 'phd': 'no'}, False),
        ({'level': 'Junior', 'lang': 'TypeScript', 'tweets': 'no', 'phd': 'yes'}, False),
        ({'level': 'Junior', 'lang': 'JavaScript', 'tweets': 'yes', 'phd': 'no'}, True),
        ({'level': 'Mid', 'lang': 'C++', 'tweets': 'yes', 'phd': 'no'}, True),
        ({'level': 'Senior', 'lang': 'Ruby', 'tweets': 'no', 'phd': 'yes'}, False),
        ({'level': 'Mid', 'lang': 'C++', 'tweets': 'no', 'phd': 'yes'}, True),
        ({'level': 'Junior', 'lang': 'Java', 'tweets': 'yes', 'phd': 'yes'}, True),
        ({'level': 'Senior', 'lang': 'Go', 'tweets': 'yes', 'phd': 'no'}, True),
        ({'level': 'Mid', 'lang': 'Go', 'tweets': 'no', 'phd': 'yes'}, True),
        ({'level': 'Junior', 'lang': 'Go', 'tweets': 'no', 'phd': 'yes'}, False),
        ({'level': 'Senior', 'lang': 'PHP', 'tweets': 'yes', 'phd': 'no'}, False),
        ({'level': 'Mid', 'lang': 'PHP', 'tweets': 'no', 'phd': 'yes'}, True),
        ({'level': 'Junior', 'lang': 'PHP', 'tweets': 'yes', 'phd': 'no'}, False),
        ({'level': 'Senior', 'lang': 'Java', 'tweets': 'yes', 'phd': 'no'}, True),
        ({'level': 'Mid', 'lang': 'Java', 'tweets': 'no', 'phd': 'no'}, False),
        ({'level': 'Junior', 'lang': 'Go', 'tweets': 'yes', 'phd': 'no'}, True),
        ({'level': 'Mid', 'lang': 'Go', 'tweets': 'yes', 'phd': 'yes'}, True),
        ({'level': 'Senior', 'lang': 'C++', 'tweets': 'no', 'phd': 'yes'}, False),
        ({'level': 'Mid', 'lang': 'JavaScript', 'tweets': 'yes', 'phd': 'no'}, True),
        ({'level': 'Junior', 'lang': 'Kotlin', 'tweets': 'no', 'phd': 'yes'}, False),
        ({'level': 'Senior', 'lang': 'TypeScript', 'tweets': 'yes', 'phd': 'yes'}, True),
        ({'level': 'Mid', 'lang': 'Haskell', 'tweets': 'no', 'phd': 'no'}, False),
        ({'level': 'Senior', 'lang': 'JavaScript', 'tweets': 'no', 'phd': 'no'}, False),
        ({'level': 'Mid', 'lang': 'Rust', 'tweets': 'yes', 'phd': 'no'}, True),
        ({'level': 'Junior', 'lang': 'Kotlin', 'tweets': 'yes', 'phd': 'no'}, True),
        ({'level': 'Mid', 'lang': 'C#', 'tweets': 'no', 'phd': 'no'}, False),
        ({'level': 'Senior', 'lang': 'PHP', 'tweets': 'no', 'phd': 'yes'}, False),
        ({'level': 'Mid', 'lang': 'Go', 'tweets': 'yes', 'phd': 'no'}, True),
        ({'level': 'Junior', 'lang': 'Elixir', 'tweets': 'yes', 'phd': 'yes'}, False),
        ({'level': 'Senior', 'lang': 'Rust', 'tweets': 'no', 'phd': 'no'}, True),
        ({'level': 'Mid', 'lang': 'TypeScript', 'tweets': 'yes', 'phd': 'yes'}, True),
        ({'level': 'Junior', 'lang': 'C++', 'tweets': 'no', 'phd': 'yes'}, False),
        ({'level': 'Senior', 'lang': 'Swift', 'tweets': 'no', 'phd': 'no'}, True),
        ({'level': 'Mid', 'lang': 'Ruby', 'tweets': 'yes', 'phd': 'yes'}, True),
        ({'level': 'Junior', 'lang': 'Elixir', 'tweets': 'no', 'phd': 'no'}, True),
        ({'level': 'Senior', 'lang': 'Haskell', 'tweets': 'yes', 'phd': 'yes'}, False),
        ({'level': 'Mid', 'lang': 'Rust', 'tweets': 'no', 'phd': 'no'}, True),
        ({'level': 'Junior', 'lang': 'PHP', 'tweets': 'no', 'phd': 'no'}, True),
        ({'level': 'Senior', 'lang': 'Java', 'tweets': 'yes', 'phd': 'yes'}, True),
        ({'level': 'Mid', 'lang': 'Python', 'tweets': 'no', 'phd': 'yes'}, True),
        ({'level': 'Junior', 'lang': 'Go', 'tweets': 'no', 'phd': 'no'}, False)
    ]

    """
    Возможные параметры для данных собеседования:
    level: ['Junior', 'Mid', 'Senior']
    lang: ['Java', 'Python', 'Kotlin', 'Swift', 'Haskell', 'C++', 'Rust', 'TypeScript', 'JavaScript', 'Go', 'PHP', 'Ruby', 'Elixir', 'C#']
    tweets: ['yes', 'no'] (Публикует ли кандидат твиты?)
    phd: ['yes', 'no'] (Есть ли у кандидата степень PhD?)
    """

    # Строим дерево решений для собеседований
    interview_tree = build_tree_id3(interview_data)

    # Классификация кандидата
    candidate = {'level': 'Junior', 'lang': 'Java', 'tweets': 'yes', 'phd': 'no'}
    print("Interview prediction:", classify(interview_tree, candidate))

    # Дерево 2: Дерево студентов
    students_data = [
        # Успешные студенты с высокими оценками, частой посещаемостью, стипендией
        ({"major": "Computer Science", "has_scholarship": "yes", "attended_events": "frequently",
          "submitted_assignments": "on_time", "athlete": "no", "teacher_feedback": "excellent",
          "hours_studied": "high", "club_member": "yes"}, True),
        ({"major": "Mathematics", "has_scholarship": "yes", "attended_events": "frequently",
          "submitted_assignments": "on_time", "athlete": "no", "teacher_feedback": "good",
          "hours_studied": "medium", "club_member": "yes"}, True),
        ({"major": "Physics", "has_scholarship": "yes", "attended_events": "frequently",
          "submitted_assignments": "on_time", "athlete": "yes", "teacher_feedback": "excellent",
          "hours_studied": "high", "club_member": "yes"}, True),
        ({"major": "Biology", "has_scholarship": "yes", "attended_events": "sometimes",
          "submitted_assignments": "on_time", "athlete": "no", "teacher_feedback": "good",
          "hours_studied": "medium", "club_member": "yes"}, True),
        ({"major": "Engineering", "has_scholarship": "yes", "attended_events": "frequently",
          "submitted_assignments": "on_time", "athlete": "no", "teacher_feedback": "excellent",
          "hours_studied": "high", "club_member": "yes"}, True),

        # Менее успешные студенты с низкой успеваемостью, реже посещающие мероприятия, без стипендии
        ({"major": "Mathematics", "has_scholarship": "no", "attended_events": "rarely",
          "submitted_assignments": "late", "athlete": "no", "teacher_feedback": "bad",
          "hours_studied": "low", "club_member": "no"}, False),
        ({"major": "Physics", "has_scholarship": "no", "attended_events": "rarely",
          "submitted_assignments": "late", "athlete": "yes", "teacher_feedback": "bad",
          "hours_studied": "low", "club_member": "no"}, False),
        ({"major": "Engineering", "has_scholarship": "no", "attended_events": "rarely",
          "submitted_assignments": "late", "athlete": "no", "teacher_feedback": "average",
          "hours_studied": "low", "club_member": "no"}, False),
        ({"major": "Computer Science", "has_scholarship": "no", "attended_events": "sometimes",
          "submitted_assignments": "late", "athlete": "no", "teacher_feedback": "bad",
          "hours_studied": "low", "club_member": "no"}, False),
        ({"major": "Chemistry", "has_scholarship": "no", "attended_events": "rarely",
          "submitted_assignments": "late", "athlete": "no", "teacher_feedback": "bad",
          "hours_studied": "low", "club_member": "no"}, False),

        # Студенты со средней успеваемостью, смешанными характеристиками
        ({"major": "Statistics", "has_scholarship": "yes", "attended_events": "sometimes",
          "submitted_assignments": "on_time", "athlete": "no", "teacher_feedback": "good",
          "hours_studied": "medium", "club_member": "yes"}, True),
        ({"major": "Biology", "has_scholarship": "no", "attended_events": "frequently",
          "submitted_assignments": "late", "athlete": "yes", "teacher_feedback": "average",
          "hours_studied": "medium", "club_member": "no"}, False),
        ({"major": "Engineering", "has_scholarship": "yes", "attended_events": "sometimes",
          "submitted_assignments": "on_time", "athlete": "yes", "teacher_feedback": "good",
          "hours_studied": "medium", "club_member": "yes"}, True),
        ({"major": "Physics", "has_scholarship": "yes", "attended_events": "frequently",
          "submitted_assignments": "on_time", "athlete": "yes", "teacher_feedback": "excellent",
          "hours_studied": "high", "club_member": "yes"}, True),
        ({"major": "Computer Science", "has_scholarship": "no", "attended_events": "sometimes",
          "submitted_assignments": "late", "athlete": "no", "teacher_feedback": "average",
          "hours_studied": "medium", "club_member": "no"}, False),

        # Различные случаи для более сложного дерева
        ({"major": "Chemistry", "has_scholarship": "yes", "attended_events": "frequently",
          "submitted_assignments": "on_time", "athlete": "no", "teacher_feedback": "excellent",
          "hours_studied": "high", "club_member": "yes"}, True),
        ({"major": "Mathematics", "has_scholarship": "no", "attended_events": "rarely",
          "submitted_assignments": "late", "athlete": "no", "teacher_feedback": "bad",
          "hours_studied": "low", "club_member": "no"}, False),
        ({"major": "Physics", "has_scholarship": "no", "attended_events": "sometimes",
          "submitted_assignments": "late", "athlete": "yes", "teacher_feedback": "average",
          "hours_studied": "low", "club_member": "no"}, False),
        ({"major": "Biology", "has_scholarship": "yes", "attended_events": "frequently",
          "submitted_assignments": "on_time", "athlete": "no", "teacher_feedback": "good",
          "hours_studied": "medium", "club_member": "yes"}, True),
        ({"major": "Engineering", "has_scholarship": "yes", "attended_events": "frequently",
          "submitted_assignments": "on_time", "athlete": "no", "teacher_feedback": "excellent",
          "hours_studied": "high", "club_member": "yes"}, True),
        ({"major": "Computer Science", "has_scholarship": "no", "attended_events": "frequently",
          "submitted_assignments": "late", "athlete": "no", "teacher_feedback": "good",
          "hours_studied": "medium", "club_member": "no"}, True),
        ({"major": "Statistics", "has_scholarship": "yes", "attended_events": "sometimes",
          "submitted_assignments": "on_time", "athlete": "no", "teacher_feedback": "average",
          "hours_studied": "medium", "club_member": "no"}, True),
        ({"major": "Mathematics", "has_scholarship": "no", "attended_events": "frequently",
          "submitted_assignments": "late", "athlete": "no", "teacher_feedback": "bad",
          "hours_studied": "low", "club_member": "no"}, False),
        ({"major": "Physics", "has_scholarship": "yes", "attended_events": "frequently",
          "submitted_assignments": "on_time", "athlete": "yes", "teacher_feedback": "excellent",
          "hours_studied": "high", "club_member": "yes"}, True),
        ({"major": "Chemistry", "has_scholarship": "no", "attended_events": "frequently",
          "submitted_assignments": "late", "athlete": "no", "teacher_feedback": "bad",
          "hours_studied": "low", "club_member": "no"}, False),
        ({"major": "Biology", "has_scholarship": "yes", "attended_events": "frequently",
          "submitted_assignments": "on_time", "athlete": "yes", "teacher_feedback": "excellent",
          "hours_studied": "high", "club_member": "yes"}, True)
    ]

    """
    Возможные параметры для данных студентов:
    major: ['Computer Science', 'Mathematics', 'Physics', 'Engineering', 'Biology', 'Chemistry', 'Statistics']
    has_scholarship: ['yes', 'no'] (Есть ли стипендия у студента?)
    attended_events: ['frequently', 'sometimes', 'rarely'] (Как часто студент посещал мероприятия?)
    submitted_assignments: ['on_time', 'late'] (Сдавал ли студент задания вовремя?)
    athlete: ['yes', 'no'] (Является ли студент спортсменом?)
    teacher_feedback: ['excellent', 'good', 'average', 'bad'] (Оценка учителя.)
    hours_studied: ['high', 'medium', 'low'] (Сколько времени студент тратит на учебу?)
    club_member: ['yes', 'no'] (Является ли студент членом клуба?)
    """

    # Строим дерево решений для студентов
    student_tree = build_tree_id3(students_data)

    # Классификация студента
    new_student = {"major": "Biology", "has_scholarship": "no", "attended_events": "rarely",
          "submitted_assignments": "late", "athlete": "yes", "teacher_feedback": "bad",
          "hours_studied": "low", "club_member": "yes"}
    print("Student prediction:", classify(student_tree, new_student))

if __name__ == "__main__":
    main()

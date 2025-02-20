import random

# DSIC Parameters
NUM_STUDENTS = 10        # total number of students (k)
EPSILON = 0.5            # acceptable error margin
P = 10                   # parameter chosen for penalty scaling
M = 20                   # maximum slope of g(x)/max penalty
LAMBDA = 2 * M / P       # lambda coefficient used in the penalty function (here, 12)
BETA = 2 * M / P         # beta coefficient used in the penalty function (here, 12)
ALPHA = 1 / NUM_STUDENTS # weighting for self-report in the aggregate score

def grading_function(x):

    if x >= 3:
        return 30 + 10 * x
    elif x >= 1:
        return 60
    else:
        return 60 * x

def compute_penalty(report_value, true_value, epsilon, lambda_val, beta, is_self):
    error = abs(report_value - true_value)
    if error <= epsilon:
        return 0
    else:
        penalty = lambda_val * (error - epsilon) ** 2
        if is_self:
            penalty += beta * report_value
        return penalty

def simulate_single_week(num_students, honesty_prob=0.5, bias_self=1.5, bias_other=1.0, seed=None):
    if seed is not None:
        random.seed(seed)
    
    true_contributions = [random.uniform(0, 10) for _ in range(num_students)]
    student_types = [random.random() < honesty_prob for _ in range(num_students)]
    reports = [[0.0 for _ in range(num_students)] for _ in range(num_students)]
    penalties = [0.0 for _ in range(num_students)]
    
    for i in range(num_students):
        for j in range(num_students):
            if student_types[i]:
                report_value = true_contributions[j]
            else:
                if i == j:
                    report_value = min(10, true_contributions[j] + bias_self)
                else:
                    report_value = max(0, true_contributions[j] - bias_other)
            reports[i][j] = report_value
            
            is_self = (i == j)
            pen = compute_penalty(report_value, true_contributions[j], EPSILON, LAMBDA, BETA, is_self)
            penalties[i] += pen
    
    aggregated = []
    for j in range(num_students):
        self_report = reports[j][j]
        others_reports = [reports[i][j] for i in range(num_students) if i != j]
        avg_others = sum(others_reports) / (num_students - 1)
        x_j = ALPHA * self_report + (1 - ALPHA) * avg_others
        aggregated.append(x_j)
    
    preliminary_grades = [grading_function(x) for x in aggregated]
    
    final_grades = [max(0, min(100, preliminary_grades[i] - penalties[i])) for i in range(num_students)]
    
    results = []
    for i in range(num_students):
        results.append({
            "student": i,
            "honest": student_types[i],
            "true_contribution": true_contributions[i],
            "self_report": reports[i][i],
            "aggregated": aggregated[i],
            "preliminary_grade": preliminary_grades[i],
            "penalty": penalties[i],
            "final_grade": final_grades[i]
        })
        
    return results

def simulate_multiple_weeks(num_iterations=1000, num_students=10, honesty_prob=0.5, bias_self=1.5, bias_other=1.0):
    """
    Run the simulation for many weeks (iterations) and compute the average final grades 
    for honest versus dishonest students.
    """
    all_results = []
    for _ in range(num_iterations):
        week_result = simulate_single_week(num_students, honesty_prob, bias_self, bias_other)
        all_results.extend(week_result)
    
    honest_grades = [res["final_grade"] for res in all_results if res["honest"]]
    dishonest_grades = [res["final_grade"] for res in all_results if not res["honest"]]
    
    avg_honest = sum(honest_grades) / len(honest_grades) if honest_grades else 0
    avg_dishonest = sum(dishonest_grades) / len(dishonest_grades) if dishonest_grades else 0
    
    return avg_honest, avg_dishonest, all_results

if __name__ == "__main__":
    NUM_ITERATIONS = 1000 # number of iterations/"weeks"
    HONESTY_PROB = 0.5    # 50% of students are honest
    BIAS_SELF = 1.5       # Dishonest students inflate their own contributions by 1.5 units
    BIAS_OTHER = 1.0      # Dishonest students deflate others' contributions by 1.0 units
    
    avg_honest, avg_dishonest, _ = simulate_multiple_weeks(
        num_iterations=NUM_ITERATIONS,
        num_students=NUM_STUDENTS,
        honesty_prob=HONESTY_PROB,
        bias_self=BIAS_SELF,
        bias_other=BIAS_OTHER
    )
    
    print("DSIC Reporting Scheme Simulation Results (over {} iterations):".format(NUM_ITERATIONS))
    print("Average final grade for HONEST students:   {:.2f}".format(avg_honest))
    print("Average final grade for DISHONEST students: {:.2f}".format(avg_dishonest))
    
    print("\nSample Report for one week:")
    sample_results = simulate_single_week(NUM_STUDENTS, HONESTY_PROB, BIAS_SELF, BIAS_OTHER, seed=42)
    for res in sample_results:
        strategy = "Honest" if res["honest"] else "Dishonest"
        print("Student {0:2d} ({1}): True Contribution = {2:5.2f}, Self Report = {3:5.2f}, Aggregated = {4:5.2f}, Preliminary Grade = {5:5.2f}, Penalty = {6:5.2f}, Final Grade = {7:5.2f}".format(
            res["student"],
            strategy,
            res["true_contribution"],
            res["self_report"],
            res["aggregated"],
            res["preliminary_grade"],
            res["penalty"],
            res["final_grade"]
        ))
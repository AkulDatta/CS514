import random
from itertools import product

# DSIC Parameters (Constants)
NUM_STUDENTS = 10        # Total number of students (k)
EPSILON = 0.9            # Acceptable error margin
P = 2.5                  # Penalty constant (must be > 2)
ALPHA = 1 / NUM_STUDENTS # Weight placed on the self-report in the aggregated score

# Grading Function:
#   if x >= 5: grade = 35 + 8 * x
#   if 2 <= x < 5: grade = 50 + 5 * x
#   if x < 2: grade = 60
def grading_function(x):
    if x >= 5:
        return 35 + 8 * x
    elif x >= 2:
        return 50 + 5 * x
    else:
        return 60

def compute_penalty(report_value, true_value, epsilon, penalty_const, is_self):
    error = abs(report_value - true_value)
    if error <= epsilon:
        return 0
    else:
        penalty = penalty_const * ((error - epsilon) ** 2)
        if is_self:
            penalty += penalty_const * report_value
        return penalty

def simulate_single_week(num_students, honesty_prob=0.5, bias_self=2, bias_other=2, seed=None):
    if seed is not None:
        random.seed(seed)
    
    true_contributions = [random.randint(0, 10) for _ in range(num_students)]
    
    student_types = [random.random() < honesty_prob for _ in range(num_students)]

    reports = [[0 for _ in range(num_students)] for _ in range(num_students)]
    penalties = [0.0 for _ in range(num_students)]
    
    for i in range(num_students):
        for j in range(num_students):
            if i == j:
                if student_types[i]:
                    report_value = true_contributions[j]
                else:
                    report_value = min(10, true_contributions[j] + bias_self)
            else:
                report_value = true_contributions[j]
            reports[i][j] = report_value
            is_self = (i == j)
            penalties[i] += compute_penalty(report_value, true_contributions[j], EPSILON, P, is_self)
    
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

def simulate_multiple_weeks(num_iterations=1000, num_students=10, honesty_prob=0.5, bias_self=2, bias_other=2):
    """
    Simulate many weeks and compute average final grades
    for honest and dishonest students.
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

def print_sample_scenario(contribution, bias_self=2, bias_other=2):
    dishonest_self = min(10, contribution + bias_self)
    penalty = compute_penalty(dishonest_self, contribution, EPSILON, P, True)
    
    honest_grade = grading_function(contribution)
    inflated_grade = grading_function(dishonest_self) - penalty
    
    print(f"True contribution: {contribution}")
    print(f"Honest peer report: {contribution}")
    print(f"Dishonest self-report (inflated): {dishonest_self}")
    print(f"Grade if reported honestly: {honest_grade:.1f}")
    print(f"Grade if self-inflated: {inflated_grade:.1f}")

def check_truth_optimality():
    """
    For every possible true contribution (v = 0 to 10) of the target student,
    and for teams of size 7 and 8, verify that reporting truthfully
    yields at least as high a final grade as reporting any inflated value.
    Peers are assumed to be accurate, meaning that each peer reports exactly v.
    """
    truth_optimal = True
    eps = 1e-9  # tolerance

    for v in range(0, 11):
        for k in [7, 8]:
            alpha_val = 1.0 / k
            avg_peer = v
            aggregated_truthful = alpha_val * v + (1 - alpha_val) * avg_peer
            truthful_grade = grading_function(aggregated_truthful)
            truthful_grade = min(100, truthful_grade)
            for r in range(v + 1, 11):
                aggregated_inflated = alpha_val * r + (1 - alpha_val) * v
                penalty = compute_penalty(r, v, EPSILON, P, True)
                inflated_grade = grading_function(aggregated_inflated) - penalty
                inflated_grade = min(100, inflated_grade)
                if inflated_grade > truthful_grade + eps:
                    print("Counterexample found:",
                          f"true value = {v}, group size = {k},",
                          f"inflated report = {r}, aggregated_inflated = {aggregated_inflated:.4f},",
                          f"inflated_grade = {inflated_grade:.4f}, truthful_grade = {truthful_grade:.4f}")
                    truth_optimal = False
    return truth_optimal

if __name__ == "__main__":
    is_truth_optimal = check_truth_optimality()
    print("\nTruth Optimality:", "True" if is_truth_optimal else "False")

    print("\nExample 1: High Contributor (8/10)")
    print_sample_scenario(8)

    print("\nExample 2: Medium Contributor (5/10)")
    print_sample_scenario(5)

    print("\nExample 3: Low Contributor (2/10)")
    print_sample_scenario(2)

    print("\nFull Simulation Results:")
    NUM_ITERATIONS = 1000 # Number of iterations (weeks)
    HONESTY_PROB = 0.5    # 50% honest students
    BIAS_SELF = 2         # Dishonest self-inflation by 2 units
    BIAS_OTHER = 2        # (Unused in this simulation)

    avg_honest, avg_dishonest, _ = simulate_multiple_weeks(
        num_iterations=NUM_ITERATIONS,
        num_students=NUM_STUDENTS,
        honesty_prob=HONESTY_PROB,
        bias_self=BIAS_SELF,
        bias_other=BIAS_OTHER
    )

    print("\nDSIC Reporting Scheme Simulation Results (over {} iterations):".format(NUM_ITERATIONS))
    print("Average final grade for HONEST students:   {:.2f}".format(avg_honest))
    print("Average final grade for DISHONEST students: {:.2f}".format(avg_dishonest))
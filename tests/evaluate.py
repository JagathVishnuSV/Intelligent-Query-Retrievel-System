import asyncio
from services.logic import answer_query
from services.doc_parser import split_into_clauses
import json

# Sample test cases with expected answers
test_cases = [
    {
        "question": "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "expected_answer_keywords": ["grace period", "premium payment", "30 days", "15 days"],
        "document_section": "This policy provides a grace period of 30 days from the due date for payment of premium. If the premium is paid within this period, the policy will continue. If the premium is not paid within the grace period, the policy will lapse."
    },
    {
        "question": "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "expected_answer_keywords": ["waiting period", "pre-existing diseases", "PED", "36 months", "three years"],
        "document_section": "Expenses related to the treatment of a Pre-Existing Disease (PED) and its direct complications shall be excluded until the expiry of thirty six (36) months of continuous coverage after the date of inception of the first policy with us."
    },
    {
        "question": "Does this policy cover maternity expenses, and what are the conditions?",
        "expected_answer_keywords": ["maternity expenses", "delivery", "pregnancy", "waiting period", "24 months"],
        "document_section": "Yes, this policy covers maternity expenses. Maternity Expenses means medical treatment expenses traceable to childbirth (including complicated deliveries and caesarean sections incurred during Hospitalization); Expenses towards lawful medical termination of pregnancy during the Policy Period. However, the Company shall not be liable to make any payment under the cover in respect of Maternity Expenses incurred in connection with or in respect of delivery or termination within a Waiting Period of twenty-four (24) months."
    }
]

async def evaluate_accuracy():
    """Evaluate the accuracy of the system using test cases."""
    correct_answers = 0
    total_tests = len(test_cases)
    
    print("Running evaluation tests...\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['question']}")
        
        # Create a mock clauses list with the document section
        clauses = [{"section": "Test Section", "text": test_case['document_section']}]
        
        try:
            # Get the answer from the system
            result = await answer_query(test_case['question'], clauses, "test_document_url")
            answer = result.answer
            
            print(f"Answer: {answer}")
            
            # Check if the answer contains expected keywords
            answer_lower = answer.lower()
            matched_keywords = [keyword for keyword in test_case['expected_answer_keywords'] if keyword.lower() in answer_lower]
            
            if len(matched_keywords) >= len(test_case['expected_answer_keywords']) * 0.5:
                correct_answers += 1
                print(f"Result: PASS (Matched keywords: {matched_keywords})")
            else:
                print(f"Result: FAIL (Matched keywords: {matched_keywords})")
                
        except Exception as e:
            print(f"Result: ERROR - {str(e)}")
            
        print("-" * 50)
    
    accuracy = (correct_answers / total_tests) * 100 if total_tests > 0 else 0
    print(f"\nEvaluation Results:")
    print(f"Correct Answers: {correct_answers}/{total_tests}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    return accuracy

def run_evaluation():
    """Run the evaluation asynchronously."""
    return asyncio.run(evaluate_accuracy())

if __name__ == "__main__":
    accuracy = run_evaluation()
    if accuracy >= 80:
        print("\nSUCCESS: System meets the 80% accuracy requirement!")
    else:
        print(f"\nNEEDS IMPROVEMENT: System is below the 80% accuracy requirement. Current accuracy: {accuracy:.2f}%")

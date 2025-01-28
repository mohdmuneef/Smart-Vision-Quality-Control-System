import logging
from datetime import datetime

# Configure logging
logging.basicConfig(filename='quality_control_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class FeedbackSystem:
    def __init__(self):
        pass

    def provide_feedback(self, products):
        feedbacks = []  # List to hold feedback for each product

        for product in products:
            product_info = product['info']
            freshness_score = product['freshness_score']
            expiry_date = product['expiry_date']
            predicted_class = product.get('predicted_class', 'Unknown')  # Default to 'Unknown' if not provided

            current_date = datetime.now().date()
            expiry_date = datetime.strptime(expiry_date, "%d/%m/%Y").date() if expiry_date else None

            # Prepare feedback messages
            feedback = f"Object Recognition: The product is identified as '{predicted_class}'. "

            # Check freshness score and expiry date
            if freshness_score < 0.5:
                feedback += f"Warning: {product_info['name']} is not fresh, reject the item."
            elif expiry_date and expiry_date < current_date:
                feedback += f"Warning: {product_info['name']} has expired, reject the item."
            else:
                feedback += f"{product_info['name']} is fresh and good for shipment."

            # Log the information
            logging.info(f"Product: {product_info['name']}, Freshness Score: {freshness_score}, "
                         f"Expiry Date: {expiry_date}, Predicted Class: {predicted_class}, Feedback: {feedback}")

            feedbacks.append(feedback)  # Add feedback to the list

        return feedbacks  # Return all feedbacks

    def log_quality_control_result(self, result):
        logging.info(f"Quality Control Result: {result}")

if __name__ == "__main__":
    # Example usage
    feedback_system = FeedbackSystem()

    # Simulate gathering predictions from the model; replace this with your actual data
    # Assuming you already have the processed images and predictions stored in a list
    processed_products = [
        {
            'info': {'name': 'Apple'},
            'freshness_score': 0.8,
            'expiry_date': "31/12/2024",
            'predicted_class': "Fresh Fruit"
        },
        {
            'info': {'name': 'Banana'},
            'freshness_score': 0.3,
            'expiry_date': "20/12/2024",
            'predicted_class': "Overripe Fruit"
        },
        {
            'info': {'name': 'cucumber'},
            'freshness_score': 0.7,
            'expiry_date': "15/12/2024",
            'predicted_class': "Fresh Vegetable"
        },
        {
            'info': {'name': 'Tomato'},
            'freshness_score': 0.6,
            'expiry_date': "10/12/2024",
            'predicted_class': "Fresh Vegetable"
        },
        {
            'info': {'name': 'okra'},
            'freshness_score': 0.4,
            'expiry_date': "05/12/2024",
            'predicted_class': "Overripe Vegetable"
        },
        {
            'info': {'name': 'Orange'},
            'freshness_score': 0.9,
            'expiry_date': "25/11/2024",
            'predicted_class': "Fresh Fruit"
        },
        {
            'info': {'name': 'Pineapple'},
            'freshness_score': 0.2,
            'expiry_date': "30/11/2024",
            'predicted_class': "Overripe Fruit"
        },
        {
            'info': {'name': 'patato'},
            'freshness_score': 0.5,
            'expiry_date': "28/11/2024",
            'predicted_class': "Fresh Vegetable"
        },
    ]

    # Call the feedback system with the processed products data
    feedbacks = feedback_system.provide_feedback(processed_products)

    # Print all feedbacks
    for feedback in feedbacks:
        print(feedback)

    # Log quality control results for each product
    for product in processed_products:
        result = {
            'product': product['info']['name'],
            'freshness_score': product['freshness_score'],
            'expiry_date': product['expiry_date'],
            'predicted_class': product.get('predicted_class', 'Unknown'),
            'feedback': feedbacks[processed_products.index(product)]
        }
        feedback_system.log_quality_control_result(result)

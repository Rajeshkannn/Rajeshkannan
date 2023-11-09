{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "# **Phase 4: Development Part 2**\n",
        "\n",
        "In this phase, we will continue building the AI project by selecting a machine learning algorithm, training the model, and evaluating its performance.\n",
        "\n",
        "*   We'll also provide code and explanations in separate cells.\n",
        "\n",
        "\n",
        " **Step 1: Selecting a Machine Learning Algorithm**\n",
        "\n",
        "\n",
        "*   For sentiment analysis, a common choice is to use a machine learning algorithm like Logistic Regression, Naive Bayes, or Support Vector Machine (SVM).\n",
        "*   In this example, we'll use Logistic Regression as the machine learning algorithm."
      ],
      "metadata": {
        "id": "7mI0nm5liSTe"
      }
    },
    {
      "cell_type": "markdown",
      "source": [],
      "metadata": {
        "id": "5WI-6E8-i4-t"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "data['airline_sentiment'] = data['airline_sentiment'].apply(lambda x: 'negative' if x == 'negative' else 'positive')\n"
      ],
      "metadata": {
        "id": "hpXzJATHgVhw"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Import necessary libraries\n",
        "import pandas as pd\n",
        "\n",
        "# Create a sample dataset with multiple sentiment classes\n",
        "data = pd.DataFrame({\n",
        "    'text': [\"I love this product!\", \"It's okay, not great.\", \"Terrible experience.\", \"Amazing!\", \"Average service.\"],\n",
        "    'sentiment': [\"positive\", \"neutral\", \"negative\", \"positive\", \"neutral\"]\n",
        "})\n",
        "\n",
        "# Verify the unique sentiment classes\n",
        "unique_classes = data['sentiment'].unique()\n",
        "print(unique_classes)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "dQ1zEaeXgr34",
        "outputId": "43f16abc-45a8-46dc-e2ab-c49ea77339a5"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "['positive' 'neutral' 'negative']\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(data.columns)\n"
      ],
      "metadata": {
        "id": "n_g-a0B8dkbT"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Import necessary libraries\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.feature_extraction.text import CountVectorizer\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.metrics import accuracy_score, classification_report\n",
        "\n",
        "# Step 1: Selecting a Machine Learning Algorithm (Logistic Regression)\n",
        "\n",
        "# Split the data into training and testing sets\n",
        "X_train, X_test, y_train, y_test = train_test_split(data['text'], data['sentiment'], test_size=0.2, random_state=42)\n",
        "\n",
        "# Convert text data to numerical features using CountVectorizer\n",
        "vectorizer = CountVectorizer()\n",
        "X_train = vectorizer.fit_transform(X_train)\n",
        "X_test = vectorizer.transform(X_test)\n",
        "# Recode 'negative' sentiment as one class, and all others as another class\n",
        "data['sentiment'] = data['sentiment'].apply(lambda x: 'negative' if x == 'negative' else 'positive')\n",
        "\n",
        "# Now, you have two classes: 'negative' and 'positive'\n",
        "\n",
        "# Initialize and train the Logistic Regression model\n",
        "model = LogisticRegression(max_iter=1000)\n",
        "model.fit(X_train, y_train)\n",
        "\n",
        "# Make predictions on the test data\n",
        "y_pred = model.predict(X_test)\n",
        "\n",
        "# Evaluate the model's performance\n",
        "accuracy = accuracy_score(y_test, y_pred)\n",
        "report = classification_report(y_test, y_pred)\n",
        "\n",
        "print(f\"Accuracy: {accuracy:.2f}\")\n",
        "print(report)\n",
        "\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "C6_0qw5adaVk",
        "outputId": "da24f467-9736-4fb0-c9a0-b27a94767d46"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Accuracy: 1.00\n",
            "              precision    recall  f1-score   support\n",
            "\n",
            "    positive       1.00      1.00      1.00         1\n",
            "\n",
            "    accuracy                           1.00         1\n",
            "   macro avg       1.00      1.00      1.00         1\n",
            "weighted avg       1.00      1.00      1.00         1\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Explanation/Documentation (Step 1):**\n",
        "\n",
        "*   We import necessary libraries including scikit-learn for machine learning.\n",
        "*   We split the data into training and testing sets using train_test_split().\n",
        "*   We convert text data into numerical features using CountVectorizer to create a Bag of Words representation.\n",
        "*   We initialize and train a Logistic Regression model.\n",
        "*   We make predictions on the test data and evaluate the model's performance using accuracy and a classification report.\n",
        "*   This code snippet demonstrates selecting a machine learning algorithm, training the model, and evaluating its performance using accuracy and a classification report.\n",
        "\n",
        "\n",
        "**Step 2: Documenting the Results**\n",
        "*   After executing the code, you can create a document summarizing the results, including accuracy, precision, recall, F1-score, and any other relevant metrics.\n",
        "*   You can also visualize the results if needed. Sharing this document is essential for assessment and documentation.\n"
      ],
      "metadata": {
        "id": "yA8b_nk-i-YN"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Import necessary libraries\n",
        "from sklearn.metrics import confusion_matrix\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "\n",
        "# Create a confusion matrix\n",
        "cm = confusion_matrix(y_test, y_pred)\n",
        "\n",
        "# Plot the confusion matrix\n",
        "plt.figure(figsize=(8, 6))\n",
        "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)\n",
        "plt.xlabel('Predicted')\n",
        "plt.ylabel('True')\n",
        "plt.title('Confusion Matrix')\n",
        "plt.show()\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 564
        },
        "id": "i8U-W5rij7YE",
        "outputId": "256bbb04-2c6e-49e4-e9e9-bfe0f352e985"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 800x600 with 2 Axes>"
            ],
            "image/png": "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nffAAw9o5syZGjZsmO666y599dVXWrFihRo1amQ3LyQkRNWqVdOiRYtUtWpVValSReHh4QoODi5XXVu2bNFrr72madOm2W5/tGTJEnXu3FnPPvusXn755XKtBwCOQmIJuLgHH3xQ+/fv10MPPaT3339fsbGxmjJlir777jvNmjVLr7zyim3um2++qRkzZmj37t0aP368tmzZooSEBP3jH/8wtaaaNWtq7dq1qly5siZNmqSlS5cqMTFRvXr1KlF7/fr19fbbbys2NlYLFixQx44dtWXLFvn5+V11/cjISG3YsEE1a9bUc889p7/97W+688479fnnn5e7KXOEp59+Wk8++aQ2btyocePGKS0tTevXr1e9evXs5lWqVElLly5VhQoVFBMTo4EDB2rbtm3leq2ffvpJw4cP1+23365nnnnGNn7PPfdo3LhxmjVrlnbs2GHK+wKAP8pilOfsdgAAAOAqSCwBAABgChpLAAAAmILGEgAAAKagsQQAAHAh27dvV69evRQUFCSLxaJ169b95vzs7GwNGjRIt956qzw8PDR+/PhS57377rtq1qyZvLy81KpVqxJfkWsYhp577jnVqVNH3t7eioyMLHFv5N9DYwkAAOBC8vPzFRoaqgULFpRpfmFhofz9/TV16lSFhoaWOueLL77QwIEDNWLECH355Zfq06eP+vTpY/ftbi+//LJeeeUVLVq0SDt37lSVKlUUFRWlCxculLl2rgoHAABwURaLRWvXrlWfPn3KNL9z584KCwsr8WUZAwYMUH5+vj766CPb2J133qmwsDAtWrRIhmEoKChITz75pCZOnChJOnv2rAICApSUlPSb30b2SySWAAAADlRYWKhz587ZHdf7W7NSU1MVGRlpNxYVFWX7yuDMzEzl5OTYzfHz81N4eHiZvlb4ipvym3fi1n7r7BIAwOXN79vc2SUALsP79jEOW3ty71qaMWOG3di0adMc+k1iv5aTk6OAgAC7sYCAAOXk5NgevzJ2tTllcVM2lgAAAK4iISFB8fHxdmNWq9VJ1TgWjSUAAIDFcWcHWq1WpzeSgYGBys3NtRvLzc1VYGCg7fErY3Xq1LGbExYWVubX4RxLAAAAi8VxhwuIiIhQcnKy3dimTZsUEREhSQoODlZgYKDdnHPnzmnnzp22OWVBYgkAAOBC8vLydOTIEdvvmZmZSk9PV40a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fVZXNj1//PFHDRkyRNWrV1flypXVvXt3HT582Pb8Y8eOqVevXqpevbqqVKmi2267TR9//LEk+63wrVu3atiwYTp79qwsFossFovtz+dfboUPGjRIAwYMsKvx4sWLqlWrlt555x1JP3+zTGJiooKDg+Xt7a3Q0FD985//dPAnBZdoLCXp8uXLWrNmjV544QW98MILWrt2Ld+4A4dZunSpqlSpop07d+rll1/WzJkztWnTJknSww8/rJMnT+pf//qX9u7dqzZt2qhr16764YcfJEkrVqzQiy++qL/85S/au3ev6tevr4ULF9qt/9NPPyk6OlopKSnasWOHmjRpoh49euinn36S9HPjKUlLlixRdna27fdf6tq1q6pVq6Y1a9bYxi5fvqxVq1Zp8ODBkqSMjAzdf//96t+/v/bv369Vq1YpJSVFY8aMMf9DA+DWli5dqooVK2rXrl2aN2+eZs+erTfffFOSNHToUO3Zs0cffPCBUlNTZRiGevTooYsXL0qSYmNjVVhYqO3bt+urr77SX/7yF/n4+JR4jbvuuktz586Vr6+vsrOzlZ2drYkTJ5aYN3jwYH344YfKy8uzjW3cuFEFBQXq27evJCkxMVHvvPOOFi1apG+++UYTJkzQY489pm3btjni48EVhgs4fPiw0aRJE6Ny5crG7bffbtx+++1G5cqVjaZNmxpHjhxxdnm4yXTq1Mm4++677cbat29vTJ482fjss88MX19f48KFC3aPh4SEGIsXLzYMwzDCw8ON2NhYu8c7dOhghIaGXvU1L1++bFStWtX48MMPbWOSjLVr19rNmzZtmt0648aNM+69917b7xs3bjSsVqvx448/GoZhGCNGjDBGjx5tt8Znn31meHh4GOfPn79qPQBQHp06dTKaN29uFBcX28YmT55sNG/e3Pj3v/9tSDI+//xz22OnT582vL29jdWrVxuGYRitWrUypk+fXuran376qSHJ9ufakiVLDD8/vxLzGjRoYMyZM8cwDMO4ePGiUatWLeOdd96xPT5w4EBjwIABhmEYxoULF4zKlSsbX3zxhd0aI0aMMAYOHFju94+yc4nEcuzYsQoJCdF//vMfpaWlKS0tTVlZWQoODtbYsWOdXR5uQq1bt7b7vU6dOjp58qT27dunvLw81axZUz4+PrYjMzNTGRkZkqRDhw7pjjvusHv+r3/Pzc3V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UJCaNm0qSfrPf/6jRx55RNWqVVONGjXUu3dvfffdd7b1Ll++rPj4eFWrVk01a9bUpEmT9OtTwH+9FV5YWKjJkyerXr16slqtaty4sd566y199913tosCq1evLovFoqFDh0qSiouLlZiYqODgYHl7eys0NFT//Oc/7V7n448/1q233ipvb2916dLFrk4AwNXRWAIwhbe3t4qKiiRJycnJOnTokDZt2qSPPvpIFy9eVFRUlKpWrarPPvtMn3/+uXx8fHT//ffbnjNr1iwlJSXp7bffVkpKin744QetXbv2N19zyJAh+vvf/65XXnlF3377rRYvXiwfHx/Vq1dPa9askSQdOnRI2dnZmjdvniQpMTFR77zzjhYtWqRvvvlGEyZM0GOPPaZt27ZJ+rkB7tevn3r16qX09HSNHDlSU6ZMcdTHBgA3FbbCAfwhhmEoOTlZGzduVFxcnE6dOqUqVarozTfftG2BL1++XMXFxXrzzTdlsVgkSUuWLFG1atW0detW3XfffZo7d64SEhLUr18/SdKiRYu0cePGq77uv//9b61evVqbNm2yfU97o0aNbI9f2TavXbu2qlWrJunnhPOll17S5s2bFRERYXtOSkqKFi9erE6dOmnhwoUKCQnRrFmzJElNmzbVV199pb/85S8mfmoAcHOisQRwTT766CP5+Pjo4sWLKi4u1qBBgzR9+nTFxsaqVatWdudV7tu3T0eOHFHVqlXt1rhw4YIyMjJ09uxZZWdnKzw83PZYxYoV1a5duxLb4Vekp6erQoUK6tSpU5lrPnLkiAoKCtStWze78aKiIt1+++2SpG+//dauDkm2JhQA8NtoLAFcky5dumjhwoXy9PRUUFCQKlb83x8nVapUsZubl5entm3basWKFSXW8ff3v6bX9/b2Lvdz8vLyJEnr169X3bp17R6zWq3XVAcA4H9oLAFckypVqqhx48ZlmtumTRutWrVKtWvXlq+vb6lz6tSpo507d6pjx46SpEuXLmnv3r1q06ZNqfNbtWql4uJibdu2zbYV/ktXEtPLly/bxlq0aCGr1aqsrKyrJp3NmzfXBx98YDe2Y8eO33+TAAAu3gHgeIMHD1atWrXUu3dvffbZZ8rMzNTWrVs1duxY/fe//5UkjRs3Tn/+85+1bt06HTx4UH/6059+8x6UDRs2VHR0tIYPH65169bZ1ly9erUkqUGDBrJYLProo4906tQp5eXlqWrVqpo4caImTJigpUuXKiMjQ2lpaZo/f76WLl0qSYqJidHhw4f11FNP6dChQ1q5cqWSkpIc/REBwE2BxhKAw1WuXFnbt29X/fr11a9fPzVv3lwjRozQhQsXbAnmk08+qccff1zR0dGKiIhQ1apV1bdv399cd+HChXrooYf0pz/9Sc2aNdOoUaOUn58vSapbt65mzJihKVOmKCAgQGPGjJEkPf/883r22WeVmJio5s2b6/7779f69esVHBwsSapfv77WrFmjdevWKTQ0VIsWLdJLL73kwE8HAG4efFc4AAAATEFiCQAAAFPQWAIAAMAUNJYAAAAwBY0lAAAATEFjCQAAAFPQWAIAAMAUNJYAAAAwBY0lAAAATEFjCQAAAFPQWAIAAMAUNJYAAAAwxf8DMdr6lTuHSd0AAAAASUVORK5CYII=\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "By adding the from sklearn.metrics import confusion_matrix import statement, you will be able to use the confusion_matrix function to create and plot the confusion matrix."
      ],
      "metadata": {
        "id": "KYhRMVPlkECU"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "RAJESH KANNAN\n",
        "\n",
        "NANDHA COLLEGE OF TECHNOLOGY\n",
        "\n",
        "26/10/2023"
      ],
      "metadata": {
        "id": "7Tf5XKW-2e_m"
      }
    }
  ]
}

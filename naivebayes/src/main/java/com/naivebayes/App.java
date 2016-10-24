package com.naivebayes;

import java.util.Map;

/**
 * Hello world!
 *
 */
public class App 
{
    public static void main( String[] args )
    {
        String testedDataset = "ionosphere.data";
        NaiveBayes nb = new NaiveBayes(testedDataset, false);

        // Print messages
        System.out.println("Testing Naive Bayes Classifier.\n");
        System.out.println("Tested Dataset: " + testedDataset);
        System.out.println();
        System.out.println("Confusion Matrix: ");

        Map<Double, Map<Double, Double>> confusionMatrix = nb.getConfusionMatrix();
        for(Double actMapKey: confusionMatrix.keySet()){
            Map<Double, Double> recordRow = confusionMatrix.get(actMapKey);
            for(Double predMapKey: recordRow.keySet()){
                Double tmpVar = recordRow.get(predMapKey);
                System.out.printf("%4d", tmpVar.intValue());
            }
            System.out.println();
        }

        // Get predicted by id
        System.out.println("\nPredicted by class: ");
        for(Double actMapKey: confusionMatrix.keySet()){
            System.out.print("Class " + actMapKey.toString() + " : ");
            System.out.println(nb.getPredictedById(actMapKey, confusionMatrix));
        }

        // Get actual by id
        System.out.println("\nActual by class: ");
        for(Double actMapKey: confusionMatrix.keySet()){
            System.out.print("Class " + actMapKey.toString() + " : ");
            System.out.println(nb.getActualById(actMapKey, confusionMatrix));
        }

        // Get Classifier Accuracy
        System.out.println("\nNaive Bayes Accuracy: " + nb.getAccuracy(confusionMatrix));
        System.out.println("\nNaive Bayes Misclassification Rate: " + nb.getMissClassRate(confusionMatrix));

        System.out.println();
        System.out.println("Recall Matrix: ");
        Map<Double, Map<Double, Double>> recallMatrix = nb.getRecall(confusionMatrix);
        for(Double actMapKey: recallMatrix.keySet()){
            Map<Double, Double> recordRow = recallMatrix.get(actMapKey);
            for(Double predMapKey: recordRow.keySet()){
                Double tmpVar = recordRow.get(predMapKey);
                System.out.print("  ");
                System.out.printf("%.2f", tmpVar);
            }
            System.out.println();
        }



    }
}

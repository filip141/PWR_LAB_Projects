package com.knnproject;

import java.util.List;
import java.util.Map;

/**
 * Hello world!
 *
 */
public class App 
{
    public static void main( String[] args ) {
        String testedDataset = "wine.data";
        KNearestNeighbours knn = new KNearestNeighbours(testedDataset, true, 12, true, true,
                "distance", "manhattan");

        // Print messages
        System.out.println("Testing ILA Classifier.\n");
        System.out.println("Tested Dataset: " + testedDataset);
        System.out.println();
        System.out.println("Confusion Matrix: ");

        Map<Double, Map<Double, Double>> confusionMatrix = LearningQuality.getConfusionMatrix(knn);
        for (Double actMapKey : confusionMatrix.keySet()) {
            Map<Double, Double> recordRow = confusionMatrix.get(actMapKey);
            for (Double predMapKey : recordRow.keySet()) {
                Double tmpVar = recordRow.get(predMapKey);
                System.out.printf("%4d", tmpVar.intValue());
            }
            System.out.println();
        }

        // Get predicted by id
        System.out.println("\nPredicted by class: ");
        for (Double actMapKey : confusionMatrix.keySet()) {
            System.out.print("Class " + actMapKey.toString() + " : ");
            System.out.println(LearningQuality.getPredictedById(actMapKey, confusionMatrix));
        }

        // Get actual by id
        System.out.println("\nActual by class: ");
        for (Double actMapKey : confusionMatrix.keySet()) {
            System.out.print("Class " + actMapKey.toString() + " : ");
            System.out.println(LearningQuality.getActualById(actMapKey, confusionMatrix));
        }

        // Get Classifier Accuracy
        System.out.println("\nILA Accuracy: " + LearningQuality.getAccuracy(confusionMatrix));
        System.out.println("\nILA Misclassification Rate: " + LearningQuality.getMissClassRate(confusionMatrix));

        double recall = 0;
        System.out.println();
        System.out.println("Recall Matrix: ");
        Map<Double, Map<Double, Double>> recallMatrix = LearningQuality.getRecall(confusionMatrix);
        for (Double actMapKey : recallMatrix.keySet()) {
            Map<Double, Double> recordRow = recallMatrix.get(actMapKey);
            for (Double predMapKey : recordRow.keySet()) {
                Double tmpVar = recordRow.get(predMapKey);
                System.out.print("  ");
                System.out.printf("%.2f", tmpVar);
                if (predMapKey.equals(actMapKey)) {
                    recall += tmpVar;
                }
            }
            System.out.println();
        }

        System.out.println();
        System.out.print("Recall: ");
        System.out.println(recall / recallMatrix.keySet().size());
        System.out.println();

        double precise = 0;
        System.out.println();
        System.out.println("Precision Matrix: ");
        Map<Double, Map<Double, Double>> precMatrix = LearningQuality.getPrecision(confusionMatrix);
        for (Double actMapKey : precMatrix.keySet()) {
            Map<Double, Double> recordRow = precMatrix.get(actMapKey);
            for (Double predMapKey : recordRow.keySet()) {
                Double tmpVar = recordRow.get(predMapKey);
                System.out.print("  ");
                System.out.printf("%.2f", tmpVar);
                if (predMapKey.equals(actMapKey)) {
                    precise += tmpVar;
                }
            }
            System.out.println();
        }
        System.out.println();
        System.out.print("Precision: ");
        System.out.println(precise / precMatrix.keySet().size());
        System.out.println();
        System.out.println("F1 score: ");
        System.out.println();
        double f1score = 0;
        List<Double> fsCore = LearningQuality.getFsCore(confusionMatrix);
        for (Double core : fsCore) {
            f1score += core;
            System.out.printf("%.2f", core);
            System.out.print("    ");
        }

        System.out.println();
        System.out.println();
        System.out.print("F1-score: ");
        System.out.println(f1score / fsCore.size());
    }
}

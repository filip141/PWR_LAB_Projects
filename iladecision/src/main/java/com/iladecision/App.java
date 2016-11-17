package com.iladecision;

import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * Hello world!
 *
 */
public class App 
{
    public static void main( String[] args )
    {

        String testedDataset = "breast_cancer.data";
        ILADecisionTree ilaTree = new ILADecisionTree(testedDataset, true,  50, true, 18, true);
        ilaTree.train();

        // Print messages
        System.out.println("Testing ILA Classifier.\n");
        System.out.println("Tested Dataset: " + testedDataset);
        System.out.println();
        System.out.println("Confusion Matrix: ");

        Map<Double, Map<Double, Double>> confusionMatrix = ilaTree.getConfusionMatrix();
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
            System.out.println(ilaTree.getPredictedById(actMapKey, confusionMatrix));
        }

        // Get actual by id
        System.out.println("\nActual by class: ");
        for(Double actMapKey: confusionMatrix.keySet()){
            System.out.print("Class " + actMapKey.toString() + " : ");
            System.out.println(ilaTree.getActualById(actMapKey, confusionMatrix));
        }

        // Get Classifier Accuracy
        System.out.println("\nILA Accuracy: " + ilaTree.getAccuracy(confusionMatrix));
        System.out.println("\nILA Misclassification Rate: " + ilaTree.getMissClassRate(confusionMatrix));

        double recall = 0;
        System.out.println();
        System.out.println("Recall Matrix: ");
        Map<Double, Map<Double, Double>> recallMatrix = ilaTree.getRecall(confusionMatrix);
        for(Double actMapKey: recallMatrix.keySet()){
            Map<Double, Double> recordRow = recallMatrix.get(actMapKey);
            for(Double predMapKey: recordRow.keySet()){
                Double tmpVar = recordRow.get(predMapKey);
                System.out.print("  ");
                System.out.printf("%.2f", tmpVar);
                if(predMapKey.equals(actMapKey)){
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
        Map<Double, Map<Double, Double>> precMatrix = ilaTree.getPrecision(confusionMatrix);
        for(Double actMapKey: precMatrix.keySet()){
            Map<Double, Double> recordRow = precMatrix.get(actMapKey);
            for(Double predMapKey: recordRow.keySet()){
                Double tmpVar = recordRow.get(predMapKey);
                System.out.print("  ");
                System.out.printf("%.2f", tmpVar);
                if(predMapKey.equals(actMapKey)){
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
        List<Double> fsCore = ilaTree.getFsCore(confusionMatrix);
        for(Double core: fsCore){
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

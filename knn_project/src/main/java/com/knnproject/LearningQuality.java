package com.knnproject;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by fbach on 1/19/2017.
 */
public class LearningQuality {

    public static Map<Double, Map<Double, Double>> getConfusionMatrix(MachineLearningAlgorithm model){

        // Variable initialization
        Double recordCounter;
        Double predictedClass;
        Map<Double, Double> tmpMapRecord;
        List<Observation> testSet;
        TrainingSet trainingSet = model.getTrainingSet();

        // Declare confusion Matrix map
        Map<Double, Map<Double, Double>> confusionMatrix = new HashMap<Double, Map<Double, Double>>();

        // Initialize confusionMatrix map
        for(Double classIterOne: trainingSet.getClasses()){
            Map<Double, Double> mapRecord = new HashMap<Double, Double>();
            for(Double classIterTwo: trainingSet.getClasses()){
                mapRecord.put(classIterTwo, 0.0);
            }
            confusionMatrix.put(classIterOne, mapRecord);
        }

        // Create confusionMatrix
        for(int tr = 0; tr < TrainingSet.dataSetParts; tr++){
            testSet = trainingSet.getTestData();
            for(Observation testedObs: testSet){
                predictedClass = (double) model.predict(testedObs.attributes);
                tmpMapRecord = confusionMatrix.get((double) testedObs.label);
                recordCounter = tmpMapRecord.get(predictedClass) + 1;
                tmpMapRecord.put(predictedClass, recordCounter);
                confusionMatrix.put((double) testedObs.label, tmpMapRecord);
            }
            // Shuffle
            model.shuffleTrainingSet();
        }

        return confusionMatrix;
    }

    public static int getPredictedById(Double keyID, Map<Double, Map<Double, Double>> confusionMatrix){
        int counter = 0;

        // Iterate over map
        for(Double actMapKey: confusionMatrix.keySet()){
            Map<Double, Double> recordRow = confusionMatrix.get(actMapKey);
            for(Double predMapKey: recordRow.keySet()){
                Double tmpVar = recordRow.get(predMapKey);
                if(predMapKey.equals(keyID)){
                    counter+=tmpVar;
                }
            }
        }
        return counter;
    }

    public static int getActualById(Double keyID, Map<Double, Map<Double, Double>> confusionMatrix){
        int counter = 0;

        // Iterate over map
        Map<Double, Double> recordRow = confusionMatrix.get(keyID);
        for(Double predMapKey: recordRow.keySet()){
            Double tmpVar = recordRow.get(predMapKey);
            counter += tmpVar;
        }
        return counter;
    }

    public static double getAccuracy(Map<Double, Map<Double, Double>> confusionMatrix){
        int counter = 0;
        double counterAll = 0;

        for(Double actMapKey: confusionMatrix.keySet()){
            Map<Double, Double> recordRow = confusionMatrix.get(actMapKey);
            Double tmpVar = recordRow.get(actMapKey);
            counter += tmpVar;
            counterAll += getActualById(actMapKey, confusionMatrix);
        }

        return counter / counterAll ;
    }

    public static double getMissClassRate(Map<Double, Map<Double, Double>> confusionMatrix){
        int counter = 0;
        double counterAll = 0;

        for(Double actMapKey: confusionMatrix.keySet()){
            Map<Double, Double> recordRow = confusionMatrix.get(actMapKey);
            for(Double predMapKey: recordRow.keySet()){
                if(!predMapKey.equals(actMapKey)){
                    Double tmpVar = recordRow.get(predMapKey);
                    counter += tmpVar;
                }
            }
            counterAll += getActualById(actMapKey, confusionMatrix);
        }

        return counter / counterAll ;
    }

    public static Map<Double, Map<Double, Double>> getRecall(Map<Double, Map<Double, Double>> confusionMatrix){

        // Declare confusion Matrix map
        double actualByClass;
        Map<Double, Map<Double, Double>> recallMatrix = new HashMap<Double, Map<Double, Double>>();

        // Initialize confusionMatrix map
        for(Double actMapKey: confusionMatrix.keySet()){
            Map<Double, Double> recordRow = confusionMatrix.get(actMapKey);
            Map<Double, Double> mapRecord = new HashMap<Double, Double>();
            actualByClass = getActualById(actMapKey, confusionMatrix);
            for(Double predMapKey: recordRow.keySet()){
                Double tmpVar = recordRow.get(predMapKey);
                mapRecord.put(predMapKey, tmpVar / actualByClass);
            }
            recallMatrix.put(actMapKey, mapRecord);
        }

        return recallMatrix;
    }

    public static Map<Double, Map<Double, Double>> getPrecision(Map<Double, Map<Double, Double>> confusionMatrix){

        // Declare confusion Matrix map
        double predByClass;
        Map<Double, Map<Double, Double>> precMatrix = new HashMap<Double, Map<Double, Double>>();

        // Initialize confusionMatrix map
        for(Double actMapKey: confusionMatrix.keySet()){
            Map<Double, Double> recordRow = confusionMatrix.get(actMapKey);
            Map<Double, Double> mapRecord = new HashMap<Double, Double>();
            for(Double predMapKey: recordRow.keySet()){
                predByClass = getPredictedById(predMapKey, confusionMatrix);
                Double tmpVar = recordRow.get(predMapKey);
                mapRecord.put(predMapKey, tmpVar / predByClass);
            }
            precMatrix.put(actMapKey, mapRecord);
        }

        return precMatrix;
    }

    public static List<Double> getFsCore(Map<Double, Map<Double, Double>> confusionMatrix){
        List<Double> fsCore = new ArrayList<Double>();
        Map<Double, Map<Double, Double>> precMatrix = getPrecision(confusionMatrix);
        Map<Double, Map<Double, Double>> recallMatrix = getRecall(confusionMatrix);

        for(Double mapKey: precMatrix.keySet()){
            Map<Double, Double> recordPrecRow = precMatrix.get(mapKey);
            Double tmpPrecVar = recordPrecRow.get(mapKey);
            Map<Double, Double> recordRecRow = recallMatrix.get(mapKey);
            Double tmpRecVar = recordRecRow.get(mapKey);

            fsCore.add(2*((tmpPrecVar * tmpRecVar) / (tmpPrecVar + tmpRecVar)));
        }
        return fsCore;
    }

}

package com.iladecision;

import java.io.IOException;
import java.util.*;

/**
 * Created by filip on 16.11.16.
 */

class PatternPair{
    public LinkedHashSet<Double> element;
    public HashSet<Integer> pattern;
    public int occurences;
    public double result;
    PatternPair(Pair pair, HashSet<Integer> pattern, Double result){
        this.element = pair.element;
        this.occurences = pair.occurences;
        this.pattern = pattern;
        this.result = result;
    }
}

class Pair{
    public LinkedHashSet<Double> element;
    public int occurences;
    Pair(LinkedHashSet<Double> element, int occurences){
        this.element = element;
        this.occurences = occurences;
    }
}

public abstract class DecisionTree {

    protected TrainingSet trainingSet;

    public DecisionTree(String dataPath, boolean classPosition, int bins, boolean equalFrequency,
                        int efRec, boolean randomCv){
        try {
            this.trainingSet =new TrainingSet(dataPath, classPosition, true, bins, equalFrequency,
                    efRec, randomCv);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public abstract int predict(List<Double> prediction);

    public abstract void shuffleTrainingSet();

    public Map<Double, Map<Double, Double>> getConfusionMatrix(){

        // Variable initialization
        Double recordCounter;
        Double predictedClass;
        Map<Double, Double> tmpMapRecord;
        List<Observation> testSet;

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
                predictedClass = (double) this.predict(testedObs.attributes);
                tmpMapRecord = confusionMatrix.get((double) testedObs.label);
                recordCounter = tmpMapRecord.get(predictedClass) + 1;
                tmpMapRecord.put(predictedClass, recordCounter);
                confusionMatrix.put((double) testedObs.label, tmpMapRecord);
            }
            // Shuffle
            shuffleTrainingSet();
        }

        return confusionMatrix;
    }

    public int getPredictedById(Double keyID, Map<Double, Map<Double, Double>> confusionMatrix){
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


    public int getActualById(Double keyID, Map<Double, Map<Double, Double>> confusionMatrix){
        int counter = 0;

        // Iterate over map
        Map<Double, Double> recordRow = confusionMatrix.get(keyID);
        for(Double predMapKey: recordRow.keySet()){
            Double tmpVar = recordRow.get(predMapKey);
            counter += tmpVar;
        }
        return counter;
    }


    public double getAccuracy(Map<Double, Map<Double, Double>> confusionMatrix){
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

    public double getMissClassRate(Map<Double, Map<Double, Double>> confusionMatrix){
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


    public Map<Double, Map<Double, Double>> getRecall(Map<Double, Map<Double, Double>> confusionMatrix){

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

    public Map<Double, Map<Double, Double>> getPrecision(Map<Double, Map<Double, Double>> confusionMatrix){

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

    public List<Double> getFsCore(Map<Double, Map<Double, Double>> confusionMatrix){
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

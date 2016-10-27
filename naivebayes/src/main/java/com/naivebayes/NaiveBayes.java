package com.naivebayes;

import java.io.IOException;
import java.util.*;

import static java.lang.Math.exp;
import static java.lang.Math.pow;
import static java.lang.Math.sqrt;

/**
 * Created by filip on 19.10.16.
 */
public abstract class NaiveBayes {

    private boolean discrete;
    private TrainingSet trainingSet;

    public NaiveBayes(String dataPath, boolean classPosition, boolean discretValues){
        try {
            this.discrete = discretValues;
            this.trainingSet =new TrainingSet(dataPath, classPosition, discretValues);
            // Normalize only for discrete values
            if(!discretValues){
                this.trainingSet.normTrainingSet();
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void shuffleTrainingSet(){
        this.trainingSet.shuffle();
        if(!discrete){
            this.trainingSet.normTrainingSet();
        }
    }

    // Calculate probability from Gaussian
    public static Double calculateGaussianProb(Double x,Double mean,Double stdev){
        Double stdPower = 2*pow(stdev, 2);
        if(!stdPower.equals(0.0)){
            Double exponent = exp(-pow(x-mean, 2) / stdPower);
            return (1 / (sqrt(2*Math.PI) * stdev)) * exponent;
        }
        return 1.0;
    }

    public int predict(List<Double> prediction){
        double[] means;
        double[] stdevs;
        double tmpPrediction;
        int finalResult = 0;
        List<Double> predictionNew = trainingSet.normalizeRecord(prediction);

        // Define predict classes and map
        Set<Double> classes = trainingSet.getClasses();
        Map<Double, Double> predictResults = new HashMap<Double, Double>();
        for(Double nbClass: classes){
            tmpPrediction = 1;
            // Define mean and std_dev
            means = TrainingSet.mean(TrainingSet.getTrainingDataByClass(nbClass.intValue(),
                    trainingSet.getTrainingData()));
            stdevs = TrainingSet.stddev(TrainingSet.getTrainingDataByClass(nbClass.intValue(),
                    trainingSet.getTrainingData()));
            for(int i = 0; i < predictionNew.size(); i++){
                tmpPrediction *= calculateGaussianProb(predictionNew.get(i), means[i], stdevs[i]);
            }
            predictResults.put(nbClass, tmpPrediction);
        }
        Double maxValueInMap = (Collections.max(predictResults.values()));
        for (Map.Entry<Double, Double> entry : predictResults.entrySet()) {
            if (entry.getValue().equals(maxValueInMap)) {
                finalResult = entry.getKey().intValue();
            }
        }

        return finalResult;
    }

    public int predictDiscrete(List<Double> prediction){

        Observation obs;
        int finalResult = 0;
        List<Observation> trainingByID;
        obs = new Observation(prediction, 0);
        obs = trainingSet.discretize(obs);

        int counterAll;
        int counterStrike;
        double tmpPrediction;
        Set<Double> classes = trainingSet.getClasses();
        Map<Double, Double> predictResults = new HashMap<Double, Double>();
        for(Double nbClass: classes){
            tmpPrediction = 1.0;
            trainingByID = TrainingSet.getTrainingDataByClass(nbClass.intValue(), trainingSet.getTrainingData());
            counterAll = trainingByID.size();
            for(int i = 0; i < obs.attributes.size(); i++){
                counterStrike = 1;
                for (Observation trainObs: trainingByID){
                    if(trainObs.attributes.get(i).equals(obs.attributes.get(i))){
                        counterStrike += 1;
                    }
                }
                tmpPrediction *= counterStrike / (1.0 * counterAll);
            }
            predictResults.put(nbClass, tmpPrediction);
        }

        Double maxValueInMap = (Collections.max(predictResults.values()));
        for (Map.Entry<Double, Double> entry : predictResults.entrySet()) {
            if (entry.getValue().equals(maxValueInMap)) {
                finalResult = entry.getKey().intValue();
            }
        }

        return finalResult;
    }

    public Map<Double, Map<Double, Double>> getConfusionMatrix(){

        // Variable initialization
        Double recordCounter;
        Double predictedClass;
        Map<Double, Double> tmpMapRecord;
        List<Observation> testSet = trainingSet.getTestData();

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
        for(int tr = 0; tr < 10; tr++){
            for(Observation testedObs: testSet){
                predictedClass = (double) this.predictDiscrete(testedObs.attributes);
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
}

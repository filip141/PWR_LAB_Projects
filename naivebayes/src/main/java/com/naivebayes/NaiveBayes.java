package com.naivebayes;

import java.io.IOException;
import java.util.*;

import static java.lang.Math.exp;
import static java.lang.Math.pow;
import static java.lang.Math.sqrt;

/**
 * Created by filip on 19.10.16.
 */
public class NaiveBayes {

    private TrainingSet trainingSet;

    public NaiveBayes(String dataPath, boolean classPosition){
        DataFile dt = new DataFile(dataPath);
        try {
            this.trainingSet = dt.getDataset(classPosition);
            this.trainingSet.normTrainingSet();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void shuffleTrainingSet(){
        this.trainingSet.shuffle();
        this.trainingSet.normTrainingSet();
    }

    // Calculate probability from Gaussian
    public static Double calculateGaussianProb(Double x,Double mean,Double stdev){
        Double exponent = exp(-pow(x-mean, 2) / (2*pow(stdev, 2)));
        return (1 / (sqrt(2*Math.PI) * stdev)) * exponent;
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
            if (entry.getValue() == maxValueInMap) {
                finalResult = entry.getKey().intValue();
            }
        }

        return finalResult;
    }

    public double testClassifier(){
        double calculatedMean;
        List<Double> meanList = new ArrayList<Double>();
        List<Observation> testSet = trainingSet.getTestData();

        for(int tr = 1; tr < 10; tr++){
            int counter = 0;
            for(Observation testedObs: testSet){
                if(this.predict(testedObs.attributes) == testedObs.label){
                    counter += 1;
                }
            }
            calculatedMean = counter * 1.0 / testSet.size();
            meanList.add(calculatedMean);

            // Shuffle
            shuffleTrainingSet();
        }

        // Sum means
        double sum= 0;
        for (Double i:meanList)
            sum = sum + i;

        return (sum / meanList.size()) * 100;
    }

}

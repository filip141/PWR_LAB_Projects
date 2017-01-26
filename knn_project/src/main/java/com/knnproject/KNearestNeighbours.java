package com.knnproject;

import java.io.IOException;
import java.util.*;

/**
 * Created by filip on 15.11.16.
 * Class representing KNN-Algorithm
 */


class ExtendedObservation {
    public Observation observation;
    public Double rate;

    public ExtendedObservation(Observation obs, Double rate){
        this.observation = obs;
        this.rate = rate;
    }

}

public class KNearestNeighbours extends MachineLearningAlgorithm {

    private int neighbours;
    private String metric;
    private String voting;
    private boolean standard;

    public KNearestNeighbours(String dataPath, boolean classPosition, int neighbours, boolean randomCv,
                              boolean standard, String voting, String metric){
        super(dataPath, classPosition, false, 20, false, 20, randomCv);
        this.neighbours = neighbours;
        this.voting = voting;
        this.metric = metric;
        this.standard = standard;
    }

    public int predict(List<Double> prediction) {
        double recordDistance;
        List<ExtendedObservation> nNeighbours;
        List<ExtendedObservation> rateList = new ArrayList<ExtendedObservation>();
        if(standard){
            prediction = trainingSet.normalizeRecord(prediction);
        }
        // Rate Training Data
        for(Observation obs: trainingSet.getTrainingData()){
            if(metric.equals("euclidean")) {
                recordDistance = MetricMethods.euclideanDistance(obs, new Observation(prediction, 0));
            }
            else{
                recordDistance = MetricMethods.manhattanDistance(obs, new Observation(prediction, 0));
            }
            rateList.add(new ExtendedObservation(obs, recordDistance));
        }

        // Sort list
        if(rateList.size() > 0) {
            Collections.sort(rateList, new Comparator<ExtendedObservation>() {
                public int compare(ExtendedObservation object1, ExtendedObservation object2) {
                    return object1.rate.compareTo(object2.rate);
                }
            });
        }

        // Get Nearest Neighbours
        nNeighbours = rateList.subList(0, neighbours);

        switch (voting) {
            case "average":
                return VotingMethods.averageDistanceVoitng(nNeighbours);

            case "majority":
                return VotingMethods.majorityVoitng(nNeighbours);

            case "distance":
                return VotingMethods.distanceVoitng(nNeighbours);
            default:
                return VotingMethods.averageDistanceVoitng(nNeighbours);
        }
    }

    public void shuffleTrainingSet() {
        this.trainingSet.shuffle();
        if(standard){
            this.trainingSet.normTrainingSet();
        }
    }

    public void train(){}
}

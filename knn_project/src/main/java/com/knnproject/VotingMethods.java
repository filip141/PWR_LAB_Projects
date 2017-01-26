package com.knnproject;

import java.util.*;

/**
 * Created by fbach on 1/19/2017.
 */
public final class VotingMethods {

    public static int majorityVoitng(List<ExtendedObservation> nNeighbours) {
        Map<Integer, Integer> classVotes = new HashMap<>();

        // Fill classes map with zeros
        for(ExtendedObservation obs: nNeighbours){
            classVotes.put(obs.observation.label, 0);
        }

        // Vote
        for(ExtendedObservation ext: nNeighbours){
            classVotes.put(ext.observation.label, classVotes.get(ext.observation.label) + 1);
        }

        return Collections.max(classVotes.entrySet(), Map.Entry.comparingByValue()).getKey();
    }


    public static int distanceVoitng(List<ExtendedObservation> nNeighbours) {
        Map<Integer, Double> classVotes = new HashMap<>();

        // Fill classes map with zeros
        for(ExtendedObservation obs: nNeighbours){
            classVotes.put(obs.observation.label, 0.0);
        }

        // Vote
        double voteValue;
        for(ExtendedObservation ext: nNeighbours){
            voteValue = classVotes.get(ext.observation.label) + 1 / ext.rate;
            classVotes.put(ext.observation.label, voteValue);
        }

        return Collections.max(classVotes.entrySet(), Map.Entry.comparingByValue()).getKey();
    }

    public static int averageDistanceVoitng(List<ExtendedObservation> nNeighbours) {
        Map<Integer, Double> classDistance = new HashMap<>();
        Map<Integer, Double> classPoints = new HashMap<>();
        Map<Integer, Double> classVotes = new HashMap<>();

        // Fill classes map with zeros
        for(ExtendedObservation obs: nNeighbours){
            classVotes.put(obs.observation.label, 0.0);
            classPoints.put(obs.observation.label, 0.0);
            classDistance.put(obs.observation.label, 0.0);
        }

        // calculate distance
        double distValue;
        for(ExtendedObservation ext: nNeighbours){
            distValue = classDistance.get(ext.observation.label) + ext.rate;
            classDistance.put(ext.observation.label, distValue);
        }

        // Count points
        double points;
        for(ExtendedObservation ext: nNeighbours){
            points = classPoints.get(ext.observation.label) + 1;
            classPoints.put(ext.observation.label, points);
        }

        // vote
        double voteValue;
        for(ExtendedObservation ext: nNeighbours){
            voteValue = classVotes.get(ext.observation.label) +
                    1/(classDistance.get(ext.observation.label) /
                            (classPoints.get(ext.observation.label)));
            classVotes.put(ext.observation.label, voteValue);
        }

        return Collections.max(classVotes.entrySet(), Map.Entry.comparingByValue()).getKey();
    }

}

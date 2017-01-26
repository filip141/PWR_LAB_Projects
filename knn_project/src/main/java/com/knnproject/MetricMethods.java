package com.knnproject;

import java.util.Iterator;

import static java.lang.Math.pow;

/**
 * Created by fbach on 1/19/2017.
 */
public final class MetricMethods {

    private MetricMethods(){}

    public static double euclideanDistance(Observation recordOne, Observation recordTwo){
        double cumsum = 0;
        Iterator<Double> it1 = recordOne.attributes.iterator();
        Iterator<Double> it2 = recordTwo.attributes.iterator();
        while(it1.hasNext() && it2.hasNext()) {
            cumsum += pow(it2.next() - it1.next(), 2);
        }

        return Math.sqrt(cumsum);
    }

    public static double manhattanDistance(Observation recordOne, Observation recordTwo){
        double cumsum = 0;
        Iterator<Double> it1 = recordOne.attributes.iterator();
        Iterator<Double> it2 = recordTwo.attributes.iterator();
        while(it1.hasNext() && it2.hasNext()) {
            cumsum += Math.abs(it2.next() - it1.next());
        }

        return Math.sqrt(cumsum);
    }
}

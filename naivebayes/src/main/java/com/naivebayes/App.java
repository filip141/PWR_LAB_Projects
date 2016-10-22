package com.naivebayes;

/**
 * Hello world!
 *
 */
public class App 
{
    public static void main( String[] args )
    {
        NaiveBayes nb = new NaiveBayes("pima-indians-diabetes.data", false);

        // Generate sample
//        List<Double> sample = new ArrayList<Double>();
//        String rawObservation = "11.82,1.47,1.99,20.8,86,1.98,1.6,.3,1.53,1.95,.95,3.33,495";
//        String[] splitedString = rawObservation.split(",");
//        for (String aSplitedString : splitedString) {
//            sample.add(Double.parseDouble(aSplitedString));
//        }
//        int finalResult = nb.predict(sample);
//        System.out.println(finalResult);
        double classifierGrade = nb.testClassifier();
        System.out.println(classifierGrade);
    }
}

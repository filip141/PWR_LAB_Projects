package com.iladecision;

/**
 * Hello world!
 *
 */
public class App 
{
    public static void main( String[] args )
    {
        String testedDataset = "wine.data";
        ILADecisionTree ilaTree = new ILADecisionTree(testedDataset, true,  50, false, 36, false);
        ilaTree.train();
        System.out.println();
    }
}

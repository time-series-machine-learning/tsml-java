package evaluation;

import evaluation.storage.ClassifierResults;

public class TestResults {

    String path = "C:\\Users\\fbu19zru\\code\\results_split\\STC\\Predictions\\";
    String datasetName = "FingerMovements";
    int dimensions = 28;

    public void showAccDimensions(){
        int dimension = 1;


        try {
            double acc = 0;
            double best = -1;
            int dimBest = 0;
            System.out.println( "Dimension Avg ");
            for (dimension = 1;dimension<=dimensions;dimension++){

                acc = getAccuracyFolds(dimension);
                System.out.println(dimension +" " + acc);
                if (acc>best){
                    best = acc;
                    dimBest = dimension;
                }
            }
            System.out.println( "Best Dimension " + dimBest +" Avg: " + best);

        } catch (Exception e) {
            e.printStackTrace();
        }

    }


    private double getAccuracyFolds(int dimension){
        ClassifierResults results = null;
        int fold = 0;
        int folds = 30;


        try {
            double accSum = 0;
            for (fold = 0;fold<folds;fold++){
                results = new ClassifierResults(path + datasetName + "Dimension" + dimension + "\\testFold"+fold+".csv");
                accSum += results.getAcc();
            }
            return accSum / (double)folds;

        } catch (Exception e) {
            e.printStackTrace();
        }
        return -1;
    }

    public static void main(String[] args){

        TestResults tr = new TestResults();
        tr.showAccDimensions();

    }

}

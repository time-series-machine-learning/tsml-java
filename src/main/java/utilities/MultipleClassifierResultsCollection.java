package utilities;

import development.DataSets;
import development.MultipleClassifierEvaluation;
import java.io.File;
import java.io.FilenameFilter;
import java.util.Arrays;
import java.util.function.Function;
import statistics.tests.TwoSampleTests;
import vector_classifiers.ChooseClassifierFromFile;
import vector_classifiers.ChooseDatasetFromFile;

/**
 * Essentially wrapper for reading in/editing a 4d list of classifier results, implemented as a list of ClassifierResultsCollection
 * convertToBigOlArray() to get into the basic ClassifierResults[][][][] format
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class MultipleClassifierResultsCollection implements DebugPrinting { 
    public String[] datasets;
    public String[] classifierNames;
    public ClassifierResultsCollection[] allResults;
    public int numFolds;

    public String baseReadPath;
    public boolean testResultsOnly;
    public boolean cleanResults;
    public boolean allowMissingResults;

    /**
     * 
     * @param classifierNames
     * @param datasets
     * @param numFolds
     * @param baseReadPath 
     * @param testResultsOnly if true, won't bother looking for/loading trainFold files
     * @param cleanResults if true, will delete the individual instance prediction info of each ClassifierResults object, 
     *      saving majority of mem requirements
     * @param allowMissingResults if false, missing fold files are reported and exception thrown, 
     *      else if true, will report results to debugprinting but just leave missing results as null.
     *      MOST other functionality here will not perform null checks. set to true mainly if just using 
     *      the batch file-reading functionality then doing your own stuff once loaded in
     * 
     * @throws Exception 
     */
    public MultipleClassifierResultsCollection(String[] classifierNames, String[] datasets, int numFolds, String baseReadPath, boolean testResultsOnly, boolean cleanResults, boolean allowMissingResults) throws Exception { 
        this.classifierNames = classifierNames;
        this.datasets = datasets;
        this.numFolds = numFolds;

        this.baseReadPath = baseReadPath;
        this.testResultsOnly = testResultsOnly;
        this.cleanResults = cleanResults;
        this.allowMissingResults = allowMissingResults;

        this.allResults = new ClassifierResultsCollection[classifierNames.length];

        ErrorReport er = new ErrorReport("Results files not found:\n");
        for (int i = 0; i < classifierNames.length; i++) {
            try {
                allResults[i] = new ClassifierResultsCollection(classifierNames[i], datasets, numFolds, baseReadPath, testResultsOnly, cleanResults, allowMissingResults);
            } catch (Exception e) {
                er.log("Classifier Errors: " + classifierNames[i] + "\n" + e);
            }
        }

        if (allowMissingResults) {
            if (!er.isEmpty())
                printlnDebug(er.getLog());
        }
        else
            er.throwIfErrors();
    }

    /**
     * Train/test is outermost array to make it easier to ignore the null train results 
     * in the case that they aren't available, i.e 
     * 
     * ClassifierResults[][][][] allRes = mcrc.convertToBigOlArray();
     * ClassifierResults[][][] testOnly = allRes[1];
     * 
     * @return ClassifierResults[train/test][classifier][dataset][fold]
     */
    public ClassifierResults[][][][] convertToBigOlArray() {
        ClassifierResults[][][][] res = new ClassifierResults[2][allResults.length][][];

        for (int i = 0; i < classifierNames.length; i++) {
            res[0][i] = allResults[i].trainResults;
            res[1][i] = allResults[i].testResults;
        }

        return res;
    }

    /**
     * Classifiers will line up with the ordering of the classifierNames field
     * 
     * @return ClassifierResults[train/test][classifier][fold]
     */
    public ClassifierResults[][][] getAllResultsForDataset(String dataset) throws Exception { 
        ClassifierResults[][][] res = new ClassifierResults[2][allResults.length][];

        int dsetIndex = Arrays.asList(datasets).indexOf(dataset);
        if (dsetIndex == -1)
            throw new Exception("(getAllResultsForDataset) Results not loaded for dataset: " + dataset);

        for (int i = 0; i < classifierNames.length; i++) {
            res[0][i] = allResults[i].trainResults[dsetIndex];
            res[1][i] = allResults[i].testResults[dsetIndex];
        }

        return res;
    }

    /**
     * @return ClassifierResults[train/test][dataset][fold]
     */
    public ClassifierResults[][][] getAllResultsForClassifier(String classifierName) throws Exception { 
        int clsIndex = Arrays.asList(classifierNames).indexOf(classifierName);
        if (clsIndex == -1)
            throw new Exception("(getAllResultsForDataset) Results not loaded for dataset: " + classifierName);

        ClassifierResults[][][] res = new ClassifierResults[][][] { allResults[clsIndex].trainResults, allResults[clsIndex].testResults };
        return res;
    }


    public double[][][][] getAccuracies() {
        return getInfo((ClassifierResults cr) -> {return cr.acc;});
    }

    public double[][][][] getInfo(Function<ClassifierResults, Double> getter) {
        ClassifierResults[][][][] allRes = convertToBigOlArray();
        double[][][][] info = new double[2][classifierNames.length][datasets.length][numFolds];
        for (int i = testResultsOnly?1:0; i < 2; i++)
            for (int j = 0; j < classifierNames.length; j++)
                for (int k = 0; k < datasets.length; k++)
                    for (int l = 0; l < numFolds; l++) 
                        info[i][j][k][l] = getter.apply(allRes[i][j][k][l]);
        return info;
    }
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    //@james, all below is duplicate code in local proj, Filtering.tests
    public static void main(String[] args) throws Exception {
//        tests1();
//        tests2();
        tests3();
    }
    
    public static  void tests3() throws Exception {
        new MultipleClassifierEvaluation( "C:/JamesLPHD/TSC_Smoothing/Analysis/", "EDvsED_FilteredFirst", 30).
            setBuildMatlabDiagrams(false).
//            setUseAllStatistics().
//            setDatasets(Arrays.copyOfRange(development.DataSets.UCIContinuousFileNames, 0, 10)). //using only 10 datasets just to make it faster... 
//            setDatasets("C:/Temp/dsets.txt").
            setDatasets(DataSets.tscProblems85).
            readInClassifiers(new String[] {"ED","ED_Filtered"}, "C:/JamesLPHD/TSC_Smoothing/Results/").
            runComparison(); 
    }
    
    public static  void tests2() throws Exception {
        String classifier = "ED";
        String baseReadPath = "C:/JamesLPHD/TSC_Smoothing/Results/";
        String[] baseDatasets = DataSets.tscProblems85;
        int numBaseDatasets = baseDatasets.length;
        int numFolds = 30;
        
        for (int dset = 0; dset < numBaseDatasets; dset++) {
            for (int fold = 0; fold < numFolds; fold++) {
                String baseDset = baseDatasets[dset];
                
                ChooseDatasetFromFile cdff = new ChooseDatasetFromFile();
                cdff.setName(classifier + "_Filtered");
                cdff.setClassifier(classifier);
                cdff.setFinalRelationName(baseDset);
                cdff.setResultsPath(baseReadPath);
                cdff.setFold(fold);
                
                String[] datasets = (new File(baseReadPath + classifier + "/Predictions/")).list(new FilenameFilter() {
                    @Override
                    public boolean accept(File dir, String name) {
                        return name.contains(baseDset);
                    }
                });
                Arrays.sort(datasets);
                if (!datasets[0].equals(baseDset))
                    throw new Exception("hwut" + baseDset  +"\n" + Arrays.toString(datasets));
                cdff.setRelationNames(datasets);
                
                cdff.buildClassifier(null);
            }
        }
    }
    
    public static void tests1() throws Exception { 
        final double P_VAL = 0.05;
        
        String baseReadPath = "C:/JamesLPHD/TSC_Smoothing/Results/";
        String[] classifiers = { "ED" };
        String[] baseDatasets = DataSets.tscProblems85;
        int numBaseDatasets = baseDatasets.length;
        int numFolds = 30;
        boolean testResultsOnly = false;
        boolean cleanResults = true;
        boolean allowMissing = false;
        
        MultipleClassifierResultsCollection[] mcrcs = new MultipleClassifierResultsCollection[numBaseDatasets];
        boolean [] aFilteredVersionIsSigBetter = new boolean[numBaseDatasets];
        boolean [] aFilteredVersionIsBetter = new boolean[numBaseDatasets];
        boolean [] unFilteredVersionIsSigBetterThanAllFiltered = new boolean[numBaseDatasets];
//        boolean [] unFilteredVersionIsBetterThanAllFiltered = new boolean[numBaseDatasets];
        
        for (int i = 0; i < numBaseDatasets; i++) {
            String datasetBase = baseDatasets[i];
            String[] datasets = (new File(baseReadPath + classifiers[0] + "/Predictions/")).list(new FilenameFilter() {
                @Override
                public boolean accept(File dir, String name) {
                    return name.contains(datasetBase);
                }
            });
            Arrays.sort(datasets);
            if (!datasets[0].equals(datasetBase))
                throw new Exception("hwut" + datasetBase  +"\n" + Arrays.toString(datasets));
            
            MultipleClassifierResultsCollection mcrc = new MultipleClassifierResultsCollection(classifiers, datasets, numFolds, baseReadPath, testResultsOnly, cleanResults, allowMissing);
            mcrcs[i] = mcrc;
            
            double[][] resFolds = mcrc.getAccuracies()[1][0]; // [test][firstclassifier]
            double[] resDsets = StatisticalUtilities.averageFinalDimension(resFolds); 
            
            double unfilteredAcc = resDsets[0];
            
            boolean allFilteredAreSigWorse = true;
            for (int j = 1; j < resDsets.length; j++) {
                double p = TwoSampleTests.studentT_PValue(resFolds[0], resFolds[j]);
                if (resDsets[j] > unfilteredAcc) {
                    aFilteredVersionIsBetter[i] = true;
                    if (p < P_VAL) 
                        aFilteredVersionIsSigBetter[i] = true;
                }
                else {
                    if (p > P_VAL)
                        allFilteredAreSigWorse = false;
                }
            }
            unFilteredVersionIsSigBetterThanAllFiltered[i] = allFilteredAreSigWorse;
        }    
        
        System.out.println("aFilteredVersionIsSigBetter: " + countNumTrue(aFilteredVersionIsSigBetter) );
        System.out.println("aFilteredVersionIsBetter: " + countNumTrue(aFilteredVersionIsBetter) );
        System.out.println("unFilteredVersionIsSigBetterThanAllFiltered: " + countNumTrue(unFilteredVersionIsSigBetterThanAllFiltered) );
    }
    
    public static int countNumTrue(boolean[] arr) { 
        int counter = 0;
        for (boolean b : arr)
            if (b) counter++;
        return counter;
    }
}
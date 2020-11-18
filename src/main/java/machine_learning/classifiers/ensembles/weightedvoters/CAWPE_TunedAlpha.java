/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package machine_learning.classifiers.ensembles.weightedvoters;

import experiments.data.DatasetLists;
import fileIO.OutFile;
import java.io.FileNotFoundException;
import java.util.concurrent.TimeUnit;

import machine_learning.classifiers.ensembles.voting.BestIndividualTrain;
import machine_learning.classifiers.ensembles.voting.MajorityConfidence;
import machine_learning.classifiers.ensembles.weightings.EqualWeighting;
import machine_learning.classifiers.ensembles.weightings.TrainAcc;
import evaluation.storage.ClassifierResults;
import utilities.StatisticalUtilities;
import machine_learning.classifiers.ensembles.CAWPE;
import weka.core.Instances;
import weka.filters.Filter;

/**
 * Tunes the value of alpha for a given dataset. Not much slower than normal hesca
 * anyway if the base classifier results are given, since we're just playing with cached results.
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class CAWPE_TunedAlpha extends CAWPE {
    //where Integer.MAX_VALUE is a marker for pick best, as alpha tends to infinity
    //0 = equal vote
    //1 = regular weighted vote
    public int[] alphaParaRange = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, Integer.MAX_VALUE }; 
    public int alpha = 4;
    public double[] alphaParaAccs = null;

    public CAWPE_TunedAlpha() { 
        super(); //sets default classifiers etc 
        
        //overwriting relevant parts 
        ensembleName = "HESCA_TunedAlpha"; 
    }   
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        printlnDebug("**HESCA TRAIN**");
        
        
        //housekeeping
        if (resultsFilesParametersInitialised) {
            if (readResultsFilesDirectories.length > 1)
                if (readResultsFilesDirectories.length != modules.length)
                    throw new Exception("HESCA.buildClassifier: more than one results path given, but number given does not align with the number of classifiers/modules.");

            if (writeResultsFilesDirectory == null)
                writeResultsFilesDirectory = readResultsFilesDirectories[0];
        }
        
        long startTime = System.nanoTime();
        
        //transform data if specified
        if(this.transform==null){
            this.trainInsts = new Instances(data);
        }else{
            this.trainInsts  = transform.transform(data);
        }
        
        //init
        this.numTrainInsts = trainInsts.numInstances();
        this.numClasses = trainInsts.numClasses();
        this.numAttributes = trainInsts.numAttributes();
        
        //set up modules
        initialiseModules();
        
        ClassifierResults[] alphaResults = new ClassifierResults[alphaParaRange.length];
        alphaParaAccs = new double[alphaParaRange.length];        
        
        double maxAcc = -1;
        int maxAccInd = -1;
        
        //in case of ties, keeps earliest intentionally, i.e favours more evenly weighted ensemble 
        //(less chance of overfitting) than going towards pick best
        for (int i = 0; i < alphaParaRange.length; i++) {
            initCombinationSchemes(alphaParaRange[i]);
            alphaResults[i] = estimateEnsemblePerformance(data); 
            alphaParaAccs[i] = alphaResults[i].getAcc();
            
            if (alphaResults[i].getAcc() > maxAcc) { 
                maxAcc = alphaResults[i].getAcc();
                maxAccInd = i;
            }
        }
        this.alpha = alphaParaRange[maxAccInd];
        initCombinationSchemes(alpha);
        trainResults = alphaResults[maxAccInd];
        
        long buildTime = System.nanoTime() - startTime;
        trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
        trainResults.setBuildTime(buildTime);
                    
        this.testInstCounter = 0; //prep for start of testing
    }
    
    protected void initCombinationSchemes(int alphaVal) throws Exception {
        if (alphaVal == 0) {
            weightingScheme = new EqualWeighting(); 
            votingScheme = new MajorityConfidence();
        }
        else if (alphaVal == Integer.MAX_VALUE) { 
            weightingScheme = new EqualWeighting(); //actual weighting is irrelevant
            votingScheme = new BestIndividualTrain(); //just copy over the results of the best individual
        } else {
            weightingScheme = new TrainAcc(alphaVal);
            votingScheme = new MajorityConfidence();
        }
        
        weightingScheme.defineWeightings(modules, numClasses);
        votingScheme.trainVotingScheme(modules, numClasses);
    }
   
    @Override
    public String getParameters(){
        StringBuilder out = new StringBuilder();
        
        if (trainResults != null) //cv performed
            out.append("BuildTime,").append(trainResults.getBuildTime()).append(",Trainacc,").append(trainResults.getAcc()).append(",");
        else 
            out.append("BuildTime,").append("-1").append(",Trainacc,").append("-1").append(",");
        
        out.append(weightingScheme.toString()).append(",").append(votingScheme.toString()).append(",");
        
        for(int m = 0; m < modules.length; m++){
            out.append(modules[m].getModuleName()).append("(").append(modules[m].priorWeight);
            for (int j = 0; j < modules[m].posteriorWeights.length; ++j)
                out.append("/").append(modules[m].posteriorWeights[j]);
            out.append("),");
        }
        
        out.append("alphaParaAccs=").append(alphaParaRange[0]).append(":").append(alphaParaAccs[0]);
        for (int i = 1; i < alphaParaRange.length; i++)
            out.append("/").append(alphaParaRange[i]).append(":").append(alphaParaAccs[i]);
                
        return out.toString();
    }
    
    
    public static void main(String[] args) throws Exception {
//        buildParaAnalysisFiles();
        
    }
    
    public static void buildParaAnalysisFiles() throws FileNotFoundException, Exception {
        String resPath = "C:/JamesLPHD/HESCA/UCI/UCIResults/";
        int numfolds = 30;
        
        String[] dsets = DatasetLists.UCIContinuousFileNames;
        String classifier = "HESCA_TunedAlpha";
        
        //both dset by fold 
        OutFile outAlphaSelected = new OutFile(resPath + classifier + "/alphaParaValues.csv");
        OutFile outDsetStdDevOverAlpha = new OutFile(resPath + classifier + "/alphaParaStdDevOverAlphaAccForEachFold.csv");
        
        //are dset (or dset_foldid) by alpha value
        OutFile outTSSAlphaAccs = new OutFile(resPath + classifier + "/alphaParaAccsByFold.csv");
        OutFile outDsetAvgAlphaAccs = new OutFile(resPath + classifier + "/alphaParasAvgOverDataset.csv");
        OutFile outDsetStdDevOverFolds = new OutFile(resPath + classifier + "/alphaParaStdDevInAccOverFoldsForEachPara.csv");
        
        
        for (int alpha : new CAWPE_TunedAlpha().alphaParaRange) {
            outTSSAlphaAccs.writeString("," + alpha);
            outDsetStdDevOverFolds.writeString("," + alpha);
            outDsetAvgAlphaAccs.writeString("," + alpha);
        }
        outTSSAlphaAccs.writeLine("");
        outDsetStdDevOverFolds.writeLine("");
        outDsetAvgAlphaAccs.writeLine("");
                       
        System.out.println("\t" + classifier);
        for (String dset : dsets) {
            System.out.println(dset);
            
            outAlphaSelected.writeString(dset);
            outDsetStdDevOverAlpha.writeString(dset);
            outDsetAvgAlphaAccs.writeString(dset);
            outDsetStdDevOverFolds.writeString(dset);
           
            double[][] alphaByFoldAccs = new double[new CAWPE_TunedAlpha().alphaParaRange.length][numfolds];
            
            for (int fold = 0; fold < numfolds; fold++) {
                
                String predictions = resPath+classifier+"/Predictions/"+dset;
                
                ClassifierResults cr = new ClassifierResults(predictions+"/testFold"+fold+".csv");
                String[] paraParts = cr.getParas().split(",");
        
                //handling outTSSAlphaAccs
                outTSSAlphaAccs.writeString(dset + "_" + fold);
                String[] alphaParaAccStrings = paraParts[paraParts.length-1].split("/");
                
                double[] alphaAccsOnThisFold = new double[alphaParaAccStrings.length];
                
                for (int i = 0; i < alphaParaAccStrings.length; ++i) {
                    String paraAcc = alphaParaAccStrings[i];
                    double acc = Double.parseDouble(paraAcc.split(":")[1]);
                    outTSSAlphaAccs.writeString("," + acc);
                    
                    alphaAccsOnThisFold[i] = acc; 
                    alphaByFoldAccs[i][fold] = acc;
                }
                outTSSAlphaAccs.writeLine("");
                
                outDsetStdDevOverAlpha.writeString("," + StatisticalUtilities.standardDeviation(alphaAccsOnThisFold, false, StatisticalUtilities.mean(alphaAccsOnThisFold, false)));
                
                //handling outAlphaSelected
                String weightToString = paraParts[2];
                if (weightToString.equals("EqualWeighting"))
                    outAlphaSelected.writeString(",equal");
                else if(weightToString.equals("BestIndividualTrain"))
                    outAlphaSelected.writeString(",pickbest");
                else {
                    int alphaSelected = 1;
                    if (weightToString.contains("("))
                        alphaSelected = (int)Double.parseDouble(weightToString.split("\\(")[1].split("\\)")[0]); // TrainAcc(4.0) => 4
                    outAlphaSelected.writeString("," + alphaSelected);

                }
            }
            
            for (int i = 0; i < alphaByFoldAccs.length; i++) {
                double meanAlphaAccOnDset = StatisticalUtilities.mean(alphaByFoldAccs[i], false);
                outDsetAvgAlphaAccs.writeString("," + meanAlphaAccOnDset);
                outDsetStdDevOverFolds.writeString("," + StatisticalUtilities.standardDeviation(alphaByFoldAccs[i], false, meanAlphaAccOnDset));
            }
            
            outAlphaSelected.writeLine("");
            outDsetStdDevOverAlpha.writeLine("");
            outDsetAvgAlphaAccs.writeLine("");
            outDsetStdDevOverFolds.writeLine("");
        }   
        
        outAlphaSelected.closeFile();
        outTSSAlphaAccs.closeFile(); 
        outDsetAvgAlphaAccs.closeFile();
        outDsetStdDevOverFolds.closeFile();
        outDsetStdDevOverAlpha.closeFile(); 
    }
}

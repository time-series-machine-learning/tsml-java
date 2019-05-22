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
package timeseriesweka.classifiers;

import fileIO.OutFile;

import java.util.*;

import timeseriesweka.classifiers.cote.HiveCoteModule;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Map.Entry;

import utilities.*;
import utilities.samplers.RandomRoundRobinIndexSampler;
import utilities.samplers.RandomStratifiedSampler;
import vector_classifiers.HomogeneousContractCAWPE;
import weka.classifiers.AbstractClassifier;
import weka.core.*;
import weka.classifiers.Classifier;
import evaluation.storage.ClassifierResults;

import static utilities.InstanceTools.resample;
import static utilities.InstanceTools.resampleTrainAndTestInstances;
import static utilities.Utilities.argMax;
import static utilities.multivariate_tools.MultivariateInstanceTools.*;
import static weka.core.Utils.sum;

/**
 * BOSS classifier with parameter search and ensembling for univariate and
 * multivariate time series classification.
 * If parameters are known, use the nested class BOSSIndividual and directly provide them.
 * 
 * Options to change the method of ensembling to randomly select parameters instead of searching.
 * Has the capability to contract train time and checkpoint when using a random ensemble.
 * 
 * Alphabetsize fixed to four and maximum wordLength of 16.
 * 
 * @author James Large, updated by Matthew Middlehurst
 * 
 * Implementation based on the algorithm described in getTechnicalInformation()
 */
public class BOSS extends AbstractClassifierWithTrainingInfo implements HiveCoteModule, TrainAccuracyEstimate, ContractClassifier, CheckpointClassifier {
    
    private int ensembleSize = 50;
    private int seed = 0;
    private int ensembleSizePerChannel = -1;
    private Random rand;
    private boolean randomEnsembleSelection = false;
    private boolean randomCVAccEnsemble = false;
    private boolean useCAWPE = false;

    private boolean useFastTrainEstimate = false;
    private int maxEvalPerClass = -1;
    private int maxEval = 500;

    private double maxWinLenProportion = 1;
    private double maxWinSearchProportion = 0.25;

    private boolean reduceTrainInstances = false;
    private double trainProportion = -1;
    private int maxTrainInstances = 1000;

    private boolean cutoff = false;
    public boolean loadAndFinish = false;

    private transient LinkedList<BOSSIndividual>[] classifiers;
    private HomogeneousContractCAWPE[] cawpe;
    private int numSeries;
    private int numClassifiers[];
    private int currentSeries = 0;
    private boolean isMultivariate = false;

    private final Integer[] wordLengths = { 16, 14, 12, 10, 8 };
    private final int alphabetSize = 4;
    private final double correctThreshold = 0.92;
    private int maxEnsembleSize = 500;
     
    private String checkpointPath;
    private String serPath;
    private boolean checkpoint = false;
    private long checkpointTime = 0;
    private long checkpointTimeDiff = 0;
    private boolean cleanupCheckpointFiles = true;
            
    private long contractTime = 0;
    private boolean contract = false;
    
    private String trainCVPath;
    private boolean trainCV = false;

    private transient Instances train;
    private double ensembleCvAcc = -1;
    private double[] ensembleCvPreds = null;

    protected static final long serialVersionUID = 22554L;

    public BOSS() {}

    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation 	result;
        result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
        result.setValue(TechnicalInformation.Field.AUTHOR, "P. Schafer");
        result.setValue(TechnicalInformation.Field.TITLE, "The BOSS is concerned with time series classification in the presence of noise");
        result.setValue(TechnicalInformation.Field.JOURNAL, "Data Mining and Knowledge Discovery");
        result.setValue(TechnicalInformation.Field.VOLUME, "29");
        result.setValue(TechnicalInformation.Field.NUMBER,"6");
        result.setValue(TechnicalInformation.Field.PAGES, "1505-1530");
        result.setValue(TechnicalInformation.Field.YEAR, "2015");

        return result;
    }

    @Override
    public Capabilities getCapabilities(){
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capabilities.Capability.RELATIONAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);

        return result;
    }

    @Override
    public String getParameters() {
        StringBuilder sb = new StringBuilder();
        sb.append(super.getParameters());

        //could be improved for different boss versions, seems low prio though

        sb.append(",numSeries,").append(numSeries);

        for (int n = 0; n < numSeries; n++) {
            sb.append(",numclassifiers").append(n).append(",").append(numClassifiers[n]);

            for (int i = 0; i < numClassifiers[n]; ++i) {
                BOSSIndividual boss = classifiers[n].get(i);
                sb.append(",windowSize,").append(boss.getWindowSize()).append(",wordLength,").append(boss.getWordLength());
                sb.append(",alphabetSize,").append(boss.getAlphabetSize()).append(",norm,").append(boss.isNorm());
            }
        }

        return sb.toString();
    }

    //set any value in nanoseconds you like.
    @Override
    public void setTimeLimit(long time){
        contractTime = time;
        contract = true;
    }

    //pass in an enum of hour, minute, day, and the amount of them.
    @Override
    public void setTimeLimit(TimeLimit time, int amount){
        switch (time){
            case DAY:
                contractTime = (long)(8.64e+13)*amount;
                break;
            case HOUR:
                contractTime = (long)(3.6e+12)*amount;
                break;
            case MINUTE:
                contractTime = (long)(6e+10)*amount;
                break;
        }
        contract = true;
    }
    
    //Set the path where checkpointed versions will be stored
    @Override
    public void setSavePath(String path){
        checkpointPath = path;
        checkpoint = true;
    }
    
    //Define how to copy from a loaded object to this object
    @Override
    public void copyFromSerObject(Object obj) throws Exception{
        if(!(obj instanceof BOSS))
            throw new Exception("The SER file is not an instance of BOSS");
        BOSS saved = ((BOSS)obj);
        System.out.println("Loading BOSS.ser");

        //copy over variables from serialised object
        ensembleSize = saved.ensembleSize;
        seed = saved.seed;
        ensembleSizePerChannel = saved.ensembleSizePerChannel;
        rand = saved.rand;
        randomEnsembleSelection = saved.randomEnsembleSelection;
        useCAWPE = saved.useCAWPE;
        cawpe = saved.cawpe;
        numSeries = saved.numSeries;
        numClassifiers = saved.numClassifiers;
        currentSeries = saved.currentSeries;
        isMultivariate = saved.isMultivariate;
        checkpointTime = saved.checkpointTime;
        cleanupCheckpointFiles = saved.cleanupCheckpointFiles;
        contractTime = saved.contractTime;
        contract = saved.contract;
        trainCVPath = saved.trainCVPath;
        trainCV = saved.trainCV;
        trainResults = saved.trainResults;
        ensembleCvAcc = saved.ensembleCvAcc;
        ensembleCvPreds = saved.ensembleCvPreds;

        //load in each serisalised classifier
        classifiers = new LinkedList[numSeries];
        for (int n = 0; n < numSeries; n++) {
            classifiers[n] = new LinkedList();
            for (int i = 0; i < saved.numClassifiers[n]; i++) {
                System.out.println("Loading BOSSIndividual" + n + "-" + i + ".ser");

                FileInputStream fis = new FileInputStream(serPath + "BOSSIndividual" + n + "-" + i + ".ser");
                try (ObjectInputStream in = new ObjectInputStream(fis)) {
                    Object indv = in.readObject();

                    if (!(indv instanceof BOSSIndividual))
                        throw new Exception("The SER file " + n + "-" + i + " is not an instance of BOSSIndividual");
                    BOSSIndividual ser = ((BOSSIndividual) indv);
                    classifiers[n].add(ser);
                }
            }
        }

        checkpointTimeDiff = saved.checkpointTimeDiff + (System.nanoTime() - checkpointTime);
    }

    @Override
    public void writeCVTrainToFile(String outputPathAndName){
        trainCVPath=outputPathAndName;
        trainCV=true;
    }

    @Override
    public void setFindTrainAccuracyEstimate(boolean setCV){
        trainCV=setCV;
    }

    @Override
    public boolean findsTrainAccuracyEstimate(){ return trainCV;}

    @Override
    public ClassifierResults getTrainResults(){
//Temporary : copy stuff into trainResults.acc here
        trainResults.setAcc(ensembleCvAcc);
//TO DO: Write the other stats
        return trainResults;
    }
    
    public void setEnsembleSize(int size) {
        ensembleSize = size;
    }

    public void setMaxEnsembleSize(int size) {
        maxEnsembleSize = size;
    }
    
    public void setSeed(int i) {
        seed = i;
    }

    public void setRandomEnsembleSelection(boolean b){
        randomEnsembleSelection = b;
    }

    public void setRandomCVAccEnsemble(boolean b){
        randomCVAccEnsemble = b;
    }

    public void useCAWPE(boolean b) {
        useCAWPE = b;
    }

    public void setFastTrainEstimate(boolean b){
        useFastTrainEstimate = b;
    }

    public void setReduceTrainInstances(boolean b){
        reduceTrainInstances = b;
    }

    public void setTrainProportion(double d){
        trainProportion = d;
    }

    public void setCleanupCheckpointFiles(boolean b) {
        cleanupCheckpointFiles = b;
    }

    public void setCutoff(boolean b) {
        cutoff = b;
    }

    public void setMaxTrainInstances(int i){
        maxTrainInstances = i;
    }

    public void setMaxEval(int i) {
        maxEval = i;
    }

    public void setMaxEvalPerClass(int i) {
        maxEvalPerClass = i;
    }

    public void setMaxWinLenProportion(double d){
        maxWinLenProportion = d;
    }

    public void setMaxWinSearchProportion(double d){
        maxWinSearchProportion = d;
    }

    @Override
    public void buildClassifier(final Instances data) throws Exception {
        trainResults.setBuildTime(System.nanoTime());

        // can classifier handle the data?
        getCapabilities().testWithFail(data);

        if(data.checkForAttributeType(Attribute.RELATIONAL)){
            isMultivariate = true;
        }

        serPath = checkpointPath + "/" + data.relationName() + seed + checkpointName() + "BOSSser/";
        File f = new File(serPath + "BOSS.ser");

        //if checkpointing and serialised files exist load said files
        if (checkpoint && f.exists()){
            long time = System.nanoTime();
            loadFromFile(serPath + "BOSS.ser");
            System.out.println("Spent " + (System.nanoTime() - time) + "nanoseconds loading files.");
        }
        //initialise variables
        else {
            if (data.classIndex() != data.numAttributes()-1)
                throw new Exception("BOSSEnsemble_BuildClassifier: Class attribute not set as last attribute in dataset");

            //Multivariate
            if (isMultivariate) {
                numSeries = numChannels(data);
                classifiers = new LinkedList[numSeries];

                for (int n = 0; n < numSeries; n++){
                    classifiers[n] = new LinkedList<>();
                }

                numClassifiers = new int[numSeries];

                if (ensembleSizePerChannel > 0){
                    ensembleSize = ensembleSizePerChannel*numSeries;
                }
            }
            //Univariate
            else{
                numSeries = 1;
                classifiers = new LinkedList[1];
                classifiers[0] = new LinkedList<>();
                numClassifiers = new int[1];
            }

            if (maxEvalPerClass > 0){
                maxEval = data.numClasses()*maxEvalPerClass;
            }

            rand = new Random(seed);
        }
        
        this.train = data;

        //required to deal with multivariate datasets, each channel is split into its own instances
        Instances[] series;

        //Multivariate
        if (isMultivariate) {
            series = splitMultivariateInstances(data);
        }
        //Univariate
        else{
            series = new Instances[1];
            series[0] = data;
        }

        //If checkpointing and flag is set stop building.
        if (!checkpoint || (checkpoint && !loadAndFinish)){
            if (randomCVAccEnsemble && useCAWPE){
                buildWeightedRandomCVAccBOSS(series);
            }
            //Contracted
            else if (contract) {
                buildContractedBOSS(series);
            }
            //Randomly selected ensemble with CAWPE weighting
            else if (useCAWPE){
                buildRandomCAWPEBOSS(series);
            }
            //Randomly selected ensemble
            else if (randomEnsembleSelection){
                buildRandomBOSS(series);
            }
            else if (randomCVAccEnsemble){
                buildRandomCVAccBOSS(series);
            }
            //Original BOSS/Accuracy cutoff ensemble
            else{
                buildBOSS(series);
            }
        }

        //end train time, converted to milliseconds currently for compatability
        trainResults.setBuildTime((System.nanoTime()/1000000) - (trainResults.getBuildTime()/1000000) - (checkpointTimeDiff/1000000));

        //Estimate train accuracy
        if (trainCV) {
            OutFile of=new OutFile(trainCVPath);
            of.writeLine(data.relationName()+",BOSSEnsemble,train");
           
            double[][] results = findEnsembleTrainAcc(data);
            of.writeLine(getParameters());
            of.writeLine(results[0][0]+"");
            ensembleCvAcc = results[0][0];
            for(int i=1;i<results[0].length;i++)
                of.writeLine(results[0][i]+","+results[1][i]);
            System.out.println("CV acc ="+results[0][0]);
            trainCV = false;
        }

        //delete any serialised files and holding folder for checkpointing on completion
        if (checkpoint && cleanupCheckpointFiles){
            checkpointCleanup();
        }
    }

    private void buildWeightedRandomCVAccBOSS(Instances[] series) throws Exception {
        int seriesLength = series[0].numAttributes()-1; //minus class attribute
        int minWindow = 10;
        int maxWindow = (int)(seriesLength*maxWinLenProportion);
        if (maxWindow < minWindow) minWindow = maxWindow/2;

        //whats the max number of window sizes that should be searched through
        double maxWindowSearches = seriesLength*maxWinSearchProportion;
        int winInc = (int)((maxWindow - minWindow) / maxWindowSearches);
        if (winInc < 1) winInc = 1;

        ArrayList<int[]>[] possibleParameters = uniqueParameters(minWindow, maxWindow, winInc);

        cawpe = new HomogeneousContractCAWPE[numSeries];
        for (int n = 0; n < numSeries; n++) {
            cawpe[n] = new HomogeneousContractCAWPE();
            cawpe[n].buildClassifier(series[n]);
            cawpe[n].setRandom(rand);
        }

        int classifiersBuilt = 0;
        double[] lowestAcc = new double[numSeries];
        for (int i = 0; i < numSeries; i++) lowestAcc[i] = Double.MAX_VALUE;

        //build classifiers up to a set size
        while (classifiersBuilt < ensembleSize && possibleParameters[numSeries-1].size() > 0) {
            int[] parameters = possibleParameters[currentSeries].remove(rand.nextInt(possibleParameters[currentSeries].size()));

            Instances data = resampleData(series[currentSeries]);

            BOSSIndividual boss = new BOSSIndividual(parameters[0], alphabetSize, parameters[1], parameters[2] == 0);
            boss.cleanAfterBuild = true;
            boss.buildClassifier(data);

            double[] classVals = data.attributeToDoubleArray(data.classIndex());
            double[][] preds = individualTrainPreds(boss, data);
            boss.accuracy = accuracy(preds, classVals);

            if(numClassifiers[currentSeries] < maxEnsembleSize){
                if (boss.accuracy < lowestAcc[currentSeries]) lowestAcc[currentSeries] = boss.accuracy;
                classifiers[currentSeries].add(boss);
                numClassifiers[currentSeries]++;
                cawpe[currentSeries].addToEnsemble(boss, preds, classVals);
            }
            else if (boss.accuracy > lowestAcc[currentSeries]) {
                classifiers[currentSeries].add(boss);
                cawpe[currentSeries].addToEnsemble(boss, preds, classVals);

                int minAccInd = (int) findMinEnsembleAcc()[0];
                classifiers[currentSeries].remove(minAccInd);
                cawpe[currentSeries].remove(minAccInd);

                lowestAcc[currentSeries] = findMinEnsembleAcc()[1];
            }

            if (isMultivariate) {
                nextSeries();
            }

            classifiersBuilt++;
        }

        if (cutoff){
            for (int n = 0; n < numSeries; n++) {
                double maxAcc = 0;
                for (int i = 0; i < classifiers[n].size(); i++){
                    if (classifiers[n].get(i).accuracy > maxAcc){
                        maxAcc = classifiers[n].get(i).accuracy;
                    }
                }

                for (int i = 0; i < classifiers[n].size(); i++){
                    BOSSIndividual b = classifiers[n].get(i);
                    if (b.accuracy < maxAcc * correctThreshold) {
                        classifiers[currentSeries].remove(i);
                        cawpe[currentSeries].remove(i);
                        numClassifiers[n]--;
                        i--;
                    }
                }
            }
        }
    }

    private void buildContractedBOSS(Instances[] series) throws Exception {
        int seriesLength = series[0].numAttributes()-1; //minus class attribute
        int minWindow = 10;
        int maxWindow = (int)(seriesLength*maxWinLenProportion);
        if (maxWindow < minWindow) minWindow = maxWindow/2;

        //whats the max number of window sizes that should be searched through
        double maxWindowSearches = seriesLength*maxWinSearchProportion;
        int winInc = (int)((maxWindow - minWindow) / maxWindowSearches);
        if (winInc < 1) winInc = 1;

        //continue building classifiers until contract time runs out or max ensemble size is reached
        //any time between runs using checkpointing is not included
        while (System.nanoTime() - trainResults.getBuildTime() - checkpointTimeDiff < contractTime && classifiers[numSeries-1].size() < maxEnsembleSize) {
            //randomly select parameters except for alphabetSize
            int wordLength = wordLengths[rand.nextInt(wordLengths.length)];
            int winSize = minWindow + winInc * rand.nextInt((int) maxWindowSearches + 1);
            if(winSize > maxWindow) winSize = maxWindow;
            boolean normalise = rand.nextBoolean();

            BOSSIndividual boss = new BOSSIndividual(wordLength, alphabetSize, winSize, normalise);
            boss.cleanAfterBuild = true;
            boss.buildClassifier(series[currentSeries]);
            classifiers[currentSeries].add(boss);
            numClassifiers[currentSeries]++;

            int prev = currentSeries;
            if (isMultivariate){
                nextSeries();
            }

            if (checkpoint) {
                checkpoint(prev);
            }
        }

        System.out.println("RBOSS Contract Data: NumClassifiers = " +
                classifiers[numSeries-1].size() + " StartTime = " + trainResults.getBuildTime() + " "
                + "EndTime = " + System.nanoTime() + " Checkpointed = " + checkpoint + " TotalTime = "
                + (System.nanoTime() - trainResults.getBuildTime() - checkpointTimeDiff) + " AverageTime = "
                + (System.nanoTime() - trainResults.getBuildTime() - checkpointTimeDiff) / classifiers[0].size());
    }

    private void buildRandomCAWPEBOSS(Instances[] series) throws Exception {
        int seriesLength = series[0].numAttributes()-1; //minus class attribute
        int minWindow = 10;
        int maxWindow = (int)(seriesLength*maxWinLenProportion);
        if (maxWindow < minWindow) minWindow = maxWindow/2;

        //whats the max number of window sizes that should be searched through
        double maxWindowSearches = seriesLength*maxWinSearchProportion;
        int winInc = (int)((maxWindow - minWindow) / maxWindowSearches);
        if (winInc < 1) winInc = 1;

        ArrayList<int[]>[] possibleParameters = uniqueParameters(minWindow, maxWindow, winInc);

        cawpe = new HomogeneousContractCAWPE[numSeries];
        for (int n = 0; n < numSeries; n++) {
            cawpe[n] = new HomogeneousContractCAWPE();
            cawpe[n].buildClassifier(series[n]);
            cawpe[n].setRandom(rand);
        }

        while (sum(numClassifiers) < ensembleSize && possibleParameters[numSeries-1].size() > 0) {
            int[] parameters = possibleParameters[currentSeries].remove(rand.nextInt(possibleParameters[currentSeries].size()));

            Instances data = resampleData(series[currentSeries]);

            BOSSIndividual boss = new BOSSIndividual(parameters[0], alphabetSize, parameters[1], parameters[2] == 0);
            boss.cleanAfterBuild = true;
            boss.buildClassifier(data);
            classifiers[currentSeries].add(boss);
            numClassifiers[currentSeries]++;

            double[] classVals = data.attributeToDoubleArray(data.classIndex());
            double[][] preds = individualTrainPreds(boss, data);

            cawpe[currentSeries].addToEnsemble(boss, preds, classVals);

            int prev = currentSeries;
            if (isMultivariate){
                nextSeries();
            }

            if (checkpoint) {
                checkpoint(prev);
            }
        }
    }

    private void buildRandomBOSS(Instances[] series) throws Exception {
        int seriesLength = series[0].numAttributes()-1; //minus class attribute
        int minWindow = 10;
        int maxWindow = (int)(seriesLength*maxWinLenProportion);
        if (maxWindow < minWindow) minWindow = maxWindow/2;

        //whats the max number of window sizes that should be searched through
        double maxWindowSearches = seriesLength*maxWinSearchProportion;
        int winInc = (int)((maxWindow - minWindow) / maxWindowSearches);
        if (winInc < 1) winInc = 1;

        //build classifiers up to a set size
        while (sum(numClassifiers) < ensembleSize) {
            //randomly select parameters except for alphabetSize
            int wordLength = wordLengths[rand.nextInt(wordLengths.length)];
            int winSize = minWindow + winInc * rand.nextInt((int) maxWindowSearches + 1);
            if(winSize > maxWindow) winSize = maxWindow;
            boolean normalise = rand.nextBoolean();

            Instances data = resampleData(series[currentSeries]);

            BOSSIndividual boss = new BOSSIndividual(wordLength, alphabetSize, winSize, normalise);
            boss.cleanAfterBuild = true;
            boss.buildClassifier(data);
            classifiers[currentSeries].add(boss);
            numClassifiers[currentSeries]++;

            int prev = currentSeries;
            if (isMultivariate){
                nextSeries();
            }

            if (checkpoint) {
                checkpoint(prev);
            }
        }
    }

    private void buildRandomCVAccBOSS(Instances[] series) throws Exception {
        int seriesLength = series[0].numAttributes()-1; //minus class attribute
        int minWindow = 10;
        int maxWindow = (int)(seriesLength*maxWinLenProportion);
        if (maxWindow < minWindow) minWindow = maxWindow/2;

        //whats the max number of window sizes that should be searched through
        double maxWindowSearches = seriesLength*maxWinSearchProportion;
        int winInc = (int)((maxWindow - minWindow) / maxWindowSearches);
        if (winInc < 1) winInc = 1;

        ArrayList<int[]>[] possibleParameters = uniqueParameters(minWindow, maxWindow, winInc);

        int classifiersBuilt = 0;
        double[] lowestAcc = new double[numSeries];
        for (int i = 0; i < numSeries; i++) lowestAcc[i] = Double.MAX_VALUE;

        //build classifiers up to a set size
        while (classifiersBuilt < ensembleSize && possibleParameters[numSeries-1].size() > 0) {
            int[] parameters = possibleParameters[currentSeries].remove(rand.nextInt(possibleParameters[currentSeries].size()));

            Instances data = resampleData(series[currentSeries]);

            BOSSIndividual boss = new BOSSIndividual(parameters[0], alphabetSize, parameters[1], parameters[2] == 0);
            boss.cleanAfterBuild = true;
            boss.buildClassifier(data);
            boss.accuracy = individualTrainAcc(boss, data, (numClassifiers[currentSeries] < maxEnsembleSize ? 0 : lowestAcc[currentSeries]));

            if(numClassifiers[currentSeries] < maxEnsembleSize){
                if (boss.accuracy < lowestAcc[currentSeries]) lowestAcc[currentSeries] = boss.accuracy;
                classifiers[currentSeries].add(boss);
                numClassifiers[currentSeries]++;
            }
            else if (boss.accuracy > lowestAcc[currentSeries]) {
                classifiers[currentSeries].add(boss);

                int minAccInd = (int) findMinEnsembleAcc()[0];
                classifiers[currentSeries].remove(minAccInd);

                lowestAcc[currentSeries] = findMinEnsembleAcc()[1];
            }

            if (isMultivariate) {
                nextSeries();
            }

            classifiersBuilt++;
        }

        if (cutoff){
            for (int n = 0; n < numSeries; n++) {
                double maxAcc = 0;
                for (int i = 0; i < classifiers[n].size(); i++){
                    if (classifiers[n].get(i).accuracy > maxAcc){
                        maxAcc = classifiers[n].get(i).accuracy;
                    }
                }

                Iterator<BOSSIndividual> it = classifiers[n].iterator();
                while (it.hasNext()) {
                    BOSSIndividual b = it.next();
                    if (b.accuracy < maxAcc * correctThreshold) {
                        it.remove();
                        numClassifiers[n]--;
                    }
                }
            }
        }
    }

    private void buildBOSS(Instances[] series) throws Exception {
        int seriesLength = series[0].numAttributes()-1; //minus class attribute
        int minWindow = 10;
        int maxWindow = (int)(seriesLength*maxWinLenProportion);
        if (maxWindow < minWindow) minWindow = maxWindow/2;

        double maxWindowSearches = seriesLength*maxWinSearchProportion;
        int winInc = (int)((maxWindow - minWindow) / maxWindowSearches);
        if (winInc < 1) winInc = 1;

        for (int n = 0; n < numSeries; n++) {
            currentSeries = n;
            double maxAcc = -1.0;

            //the acc of the worst member to make it into the final ensemble as it stands
            double minMaxAcc = -1.0;

            boolean[] normOptions = {true, false};

            for (boolean normalise : normOptions) {
                for (int winSize = minWindow; winSize <= maxWindow; winSize += winInc) {
                    BOSSIndividual boss = new BOSSIndividual(wordLengths[0], alphabetSize, winSize, normalise);
                    boss.buildClassifier(series[n]); //initial setup for this windowsize, with max word length

                    BOSSIndividual bestClassifierForWinSize = null;
                    double bestAccForWinSize = -1.0;

                    //find best word length for this window size
                    for (Integer wordLen : wordLengths) {
                        boss = boss.buildShortenedBags(wordLen); //in first iteration, same lengths (wordLengths[0]), will do nothing

                        double acc = individualTrainAcc(boss, series[n], bestAccForWinSize);

                        if (acc >= bestAccForWinSize) {
                            bestAccForWinSize = acc;
                            bestClassifierForWinSize = boss;
                        }
                    }

                    //if this window size's accuracy is not good enough to make it into the ensemble, dont bother storing at all
                    if (makesItIntoEnsemble(bestAccForWinSize, maxAcc, minMaxAcc, classifiers[n].size())) {
                        bestClassifierForWinSize.clean();
                        bestClassifierForWinSize.accuracy = bestAccForWinSize;
                        classifiers[n].add(bestClassifierForWinSize);

                        if (bestAccForWinSize > maxAcc) {
                            maxAcc = bestAccForWinSize;
                            //get rid of any extras that dont fall within the new max threshold
                            Iterator<BOSSIndividual> it = classifiers[n].iterator();
                            while (it.hasNext()) {
                                BOSSIndividual b = it.next();
                                if (b.accuracy < maxAcc * correctThreshold) {
                                    it.remove();
                                }
                            }
                        }

                        while (classifiers[n].size() > maxEnsembleSize) {
                            //cull the 'worst of the best' until back under the max size
                            int minAccInd = (int) findMinEnsembleAcc()[0];

                            classifiers[n].remove(minAccInd);
                        }

                        minMaxAcc = findMinEnsembleAcc()[1]; //new 'worst of the best' acc
                    }

                    numClassifiers[n] = classifiers[n].size();
                }
            }
        }
    }
    
    private void checkpoint(int seriesNo){
        if(checkpointPath!=null){
            try{
                File f = new File(serPath);
                if(!f.isDirectory())
                    f.mkdirs();
                //time the checkpoint occured
                checkpointTime = System.nanoTime();

                if (seriesNo >= 0) {
                    //save the last build individual classifier
                    BOSSIndividual indiv = classifiers[seriesNo].get(classifiers[seriesNo].size() - 1);

                    FileOutputStream fos = new FileOutputStream(serPath + "BOSSIndividual" + seriesNo + "-" + (classifiers[seriesNo].size() - 1) + ".ser");
                    try (ObjectOutputStream out = new ObjectOutputStream(fos)) {
                        out.writeObject(indiv);
                        out.close();
                        fos.close();
                    }
                }

                //dont take into account time spent serialising into build time
                checkpointTimeDiff += System.nanoTime() - checkpointTime;
                checkpointTime = System.nanoTime();

                //save this, saved classifiers not included
                saveToFile(serPath + "RandomBOSStemp.ser");

                File file = new File(serPath + "RandomBOSStemp.ser");
                File file2 = new File(serPath + "BOSS.ser");
                file2.delete();
                file.renameTo(file2);
            }
            catch(Exception e){
                e.printStackTrace();
                System.out.println("Serialisation to "+serPath+" FAILED");
            }
        }
    }

    public void checkpointCleanup(){
        File f = new File(serPath);
        String[] files = f.list();

        for (String file: files){
            File f2 = new File(f.getPath() + "\\" + file);
            boolean b = f2.delete();
        }

        f.delete();
    }

    public String checkpointName(){
        if (contract){
            return "RandomContract" + contractTime;
        }
        else if (useCAWPE){
            if (isMultivariate && ensembleSizePerChannel > 0){
                return "RandomCAWPE" + (ensembleSizePerChannel*numSeries);
            }
            else {
                return "RandomCAWPE" + ensembleSize;
            }
        }
        else if (randomEnsembleSelection){
            if (isMultivariate && ensembleSizePerChannel > 0){
                return "Random" + (ensembleSizePerChannel*numSeries);
            }
            else {
                return "Random" + ensembleSize;
            }
        }

        return "";
    }

    //[0] = index, [1] = acc
    private double[] findMinEnsembleAcc() {
        double minAcc = Double.MAX_VALUE;
        int minAccInd = 0;
        for (int i = 0; i < classifiers[currentSeries].size(); ++i) {
            double curacc = classifiers[currentSeries].get(i).accuracy;
            if (curacc < minAcc) {
                minAcc = curacc;
                minAccInd = i;
            }
        }

        return new double[] { minAccInd, minAcc };
    }

    private ArrayList<int[]>[] uniqueParameters(int minWindow, int maxWindow, int winInc){
        ArrayList<int[]>[] possibleParameters = new ArrayList[numSeries];

        for (int n = 0; n < numSeries; n++) {
            possibleParameters[n] = new ArrayList<>();

            for (int normalise = 0; normalise < 2; normalise++) {
                for (int winSize = minWindow; winSize <= maxWindow; winSize += winInc) {
                    for (Integer wordLen : wordLengths) {
                        int[] parameters = {wordLen, winSize, normalise};
                        possibleParameters[n].add(parameters);
                    }
                }
            }
        }

        return possibleParameters;
    }

    private Instances resampleData(Instances series){
        Instances data;

        if (reduceTrainInstances && trainProportion > 0){
            data = resample(series, trainProportion, rand);
        }
        else if (reduceTrainInstances && series.numInstances() > maxTrainInstances) {
            RandomStratifiedSampler sampler = new RandomStratifiedSampler(rand);
            sampler.setInstances(series);
            data = new Instances(series, 0);

            for (int i = 0; i < maxTrainInstances; i++){
                data.add(sampler.next());
            }

            //System.out.println(data.numInstances() + " " + Arrays.toString(classDistribution(data)) + " " + Arrays.toString(classDistribution(series)));
        }
        else{
            data = series;
        }

        return data;
    }

    private double individualTrainAcc(BOSSIndividual boss, Instances series, double lowestAcc) throws Exception {
        if (useFastTrainEstimate && maxEval < series.numInstances()){
            RandomRoundRobinIndexSampler sampler = new RandomRoundRobinIndexSampler(rand);
            sampler.setInstances(series);

            int correct = 0;
            int requiredCorrect = (int)lowestAcc*maxEval;

            for (int i = 0; i < maxEval; ++i) {
                if (correct + maxEval - i < requiredCorrect){
                    return -1;
                }

                int subsampleIndex = sampler.next();
                double c = boss.classifyInstance(subsampleIndex); //classify series subsampleIndex, while ignoring its corresponding histogram subsampleIndex
                if (c == series.get(subsampleIndex).classValue())
                    ++correct;
            }

            return (double) correct / (double) maxEval;
        }
        else {
            int correct = 0;
            int numInst = series.numInstances();
            int requiredCorrect = (int)lowestAcc*numInst;

            for (int i = 0; i < numInst; ++i) {
                if (correct + numInst - i < requiredCorrect){
                    return -1;
                }

                double c = boss.classifyInstance(i); //classify series i, while ignoring its corresponding histogram i
                if (c == series.get(i).classValue())
                    ++correct;
            }

            return (double) correct / (double) numInst;
        }
    }

    private double[][] individualTrainPreds(BOSSIndividual boss, Instances series) throws Exception {
        int subsampleSize = series.numClasses()* maxEvalPerClass;
        double[][] preds;

        if (useFastTrainEstimate && subsampleSize < series.numInstances()){
            RandomRoundRobinIndexSampler sampler = new RandomRoundRobinIndexSampler(rand);
            sampler.setInstances(series);
            preds = new double[subsampleSize][series.numClasses()];

            for (int i = 0; i < subsampleSize; ++i) {
                int subsampleIndex = sampler.next();
                double c = boss.classifyInstance(subsampleIndex); //classify series subsampleIndex, while ignoring its corresponding histogram subsampleIndex
                preds[i][(int)c] = 1.0;
            }
        }
        else {
            int numInst = series.numInstances();
            preds = new double[numInst][series.numClasses()];

            for (int i = 0; i < numInst; ++i) {
                double c = boss.classifyInstance(i); //classify series i, while ignoring its corresponding histogram i
                preds[i][(int)c] = 1.0;
            }
        }

        return preds;
    }

    public double accuracy(double[][] probs, double[] classVals){
        double correct = 0;
        for (int i = 0; i < classVals.length; i++){
            if (argMax(probs[i], rand) == classVals[i]){
                correct++;
            }
        }
        return correct/classVals.length;
    }

    private boolean makesItIntoEnsemble(double acc, double maxAcc, double minMaxAcc, int curEnsembleSize) {
        if (acc >= maxAcc * correctThreshold) {
            if (curEnsembleSize >= maxEnsembleSize)
                return acc > minMaxAcc;
            else
                return true;
        }

        return false;
    }

    public void nextSeries(){
        if (currentSeries == numSeries-1){
            currentSeries = 0;
        }
        else{
            currentSeries++;
        }
    }

    private double[][] findEnsembleTrainAcc(Instances data) throws Exception {
        if (useCAWPE || (reduceTrainInstances && (data.numInstances() > maxTrainInstances || trainProportion > 0))){
            throw new Exception("Unable to create ensemble train estimate for these settings");
        }
        else {
            double[][] results = new double[2 + data.numClasses()][data.numInstances() + 1];

            this.ensembleCvPreds = new double[data.numInstances()];

            double correct = 0;
            for (int i = 0; i < data.numInstances(); ++i) {
                double[] probs = distributionForInstance(i, data.numClasses());


                double c = 0;
                for (int j = 1; j < probs.length; j++)
                    if (probs[j] > probs[(int) c])
                        c = j;
                //No need to do it againclassifyInstance(i, data.numClasses()); //classify series i, while ignoring its corresponding histogram i
                if (c == data.get(i).classValue())
                    ++correct;
                results[0][i + 1] = data.get(i).classValue();
                results[1][i + 1] = c;
                for (int j = 0; j < probs.length; j++)
                    results[2 + j][i + 1] = probs[j];
                this.ensembleCvPreds[i] = c;
            }

            results[0][0] = correct / data.numInstances();
            //TODO fill results[1][0]

            return results;
        }
    }
    
    public double getEnsembleCvAcc(){
        if(ensembleCvAcc>=0){
            return this.ensembleCvAcc;
        }
        
        try{
            return this.findEnsembleTrainAcc(train)[0][0];
        }catch(Exception e){
            e.printStackTrace();
        }
        return -1;
    }
    
    public double[] getEnsembleCvPreds(){
        if(this.ensembleCvPreds==null){   
            try{
                this.findEnsembleTrainAcc(train);
            }catch(Exception e){
                e.printStackTrace();
            }
        }
        
        return this.ensembleCvPreds;
    }

    private double[] distributionForInstance(int test, int numclasses) throws Exception {
        double[][] classHist = new double[numSeries][numclasses];

        //get sum of all channels, votes from each are weighted the same.
        double sum[] = new double[numSeries];

        for (int n = 0; n < numSeries; n++) {
            for (BOSSIndividual classifier : classifiers[n]) {
                double classification = classifier.classifyInstance(test);
                classHist[n][(int) classification]++;
                sum[n]++;
            }
        }

        double[] distributions = new double[numclasses];

        for (int n = 0; n < numSeries; n++){
            if (sum[n] != 0)
                for (int i = 0; i < classHist[n].length; ++i)
                    distributions[i] += (classHist[n][i] / sum[n]) / numSeries;
        }

        return distributions;
    }
    
    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double[] dist = distributionForInstance(instance);
        
        double maxFreq=dist[0], maxClass=0;
        for (int i = 1; i < dist.length; ++i) 
            if (dist[i] > maxFreq) {
                maxFreq = dist[i];
                maxClass = i;
            }
        
        return maxClass;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        double[][] classHist = new double[numSeries][instance.numClasses()];

        //get sum of all channels, votes from each are weighted the same.
        double sum[] = new double[numSeries];

        Instance[] series;

        //Multivariate
        if (isMultivariate) {
            series = splitMultivariateInstanceWithClassVal(instance);
        }
        //Univariate
        else {
            series = new Instance[1];
            series[0] = instance;
        }

        if (cawpe == null) {
            for (int n = 0; n < numSeries; n++) {
                for (BOSSIndividual classifier : classifiers[n]) {
                    double classification = classifier.classifyInstance(series[n]);
                    classHist[n][(int) classification]++;
                    sum[n]++;
                }
            }
        }
        //Special case for CAWPE
        else {
            for (int n = 0; n < numSeries; n++) {
                double[] dist = cawpe[n].distributionForInstance(series[n]);

                for (int i = 0; i < dist.length; i++) {
                    classHist[n][i] += dist[i];
                }

                sum[n]++;
            }
        }

        double[] distributions = new double[instance.numClasses()];

        for (int n = 0; n < numSeries; n++){
            if (sum[n] != 0)
                for (int i = 0; i < classHist[n].length; ++i)
                    distributions[i] += (classHist[n][i] / sum[n]) / numSeries;
        }

        return distributions;
    }

    public static void main(String[] args) throws Exception{
        int fold = 1;

        //Minimum working example
        String dataset = "SonyAIBORobotSurface1";
        Instances train = ClassifierTools.loadData("Z:\\Data\\TSCProblems2018\\"+dataset+"\\"+dataset+"_TRAIN.arff");
        Instances test = ClassifierTools.loadData("Z:\\Data\\TSCProblems2018\\"+dataset+"\\"+dataset+"_TEST.arff");
        Instances[] data = resampleTrainAndTestInstances(train, test, fold);
        train = data[0];
        test = data[1];

        String dataset2 = "PenDigits";
        Instances train2 = ClassifierTools.loadData("Z:\\Data\\MultivariateTSCProblems\\"+dataset2+"\\"+dataset2+"_TRAIN.arff");
        Instances test2 = ClassifierTools.loadData("Z:\\Data\\MultivariateTSCProblems\\"+dataset2+"\\"+dataset2+"_TEST.arff");

        Classifier c;
        double accuracy;

//                c = new BOSS();
//        c.buildClassifier(train);
//        accuracy = ClassifierTools.accuracy(test, c);
//
//        System.out.println("BOSS accuracy on " + dataset + " fold 0 = " + accuracy);

        c = new BOSS();
        ((BOSS) c).setEnsembleSize(250);
        ((BOSS) c).setMaxEnsembleSize(50);
        ((BOSS) c).setRandomCVAccEnsemble(true);
        ((BOSS) c).useCAWPE(true);
        ((BOSS) c).setSeed(fold);
        ((BOSS) c).setReduceTrainInstances(true);
        ((BOSS) c).setTrainProportion(0.7);
        c.buildClassifier(train);
        accuracy = ClassifierTools.accuracy(test, c);

        System.out.println("CAWPE BOSS accuracy on " + dataset + " fold 0 = " + accuracy);

//        c = new BOSS();
//
//        ((BOSS) c).ensembleSize = 250;
//        ((BOSS) c).setMaxEnsembleSize(50);
//        ((BOSS) c).setRandomCVAccEnsemble(true);
//        ((BOSS) c).setSeed(0);
//        ((BOSS) c).useFastTrainEstimate = true;
//        ((BOSS) c).setMaxEvalPerClass(50);
//        //((BOSS) c).reduceTrainInstances = true;
//        //((BOSS) c).cutoff = true;
//        c.buildClassifier(train);
//        accuracy = ClassifierTools.accuracy(test, c);
//
//        System.out.println("Random CV Acc BOSS accuracy on " + dataset + " fold 0 = " + accuracy);

////        c = new BOSS();
////        ((BOSS) c).ensembleSize = 250;
////        ((BOSS) c).randomEnsembleSelection = true;
////        ((BOSS) c).setSavePath("C:\\UEAMachineLearning");
////        c.buildClassifier(train2);
////        accuracy = ClassifierTools.accuracy(test2, c);
////
////        System.out.println(((BOSS) c).numSeries);
////        System.out.println(Arrays.toString(((BOSS) c).numClassifiers));
////
////        System.out.println("Random BOSS MV accuracy on " + dataset2 + " fold 0 = " + accuracy);
////
//        c = new BOSS();
//        ((BOSS) c).ensembleSize = 100;
//        ((BOSS) c).randomEnsembleSelection = true;
//        ((BOSS) c).setSeed(0);
//        ((BOSS) c).setReduceTrainInstances(true);
//        ((BOSS) c).setTrainProportion(0.7);
//        c.buildClassifier(train);
//        accuracy = ClassifierTools.accuracy(test, c);
//
//        System.out.println("Random BOSS accuracy on " + dataset + " fold 0 = " + accuracy);
////
////        c = new BOSS();
////        ((BOSS) c).ensembleSize = 250;
////        ((BOSS) c).useCAWPE = true;
////        ((BOSS) c).setSavePath("C:\\UEAMachineLearning");
////        c.buildClassifier(train2);
////        accuracy = ClassifierTools.accuracy(test2, c);
////
////        System.out.println(((BOSS) c).numSeries);
////        System.out.println(Arrays.toString(((BOSS) c).numClassifiers));
////
////        System.out.println("CAWPE BOSS MV accuracy on " + dataset2 + " fold 0 = " + accuracy);
////
//        c = new BOSS();
//        ((BOSS) c).ensembleSize = 250;
//        ((BOSS) c).useCAWPE = true;
//        ((BOSS) c).setSeed(0);
//        c.buildClassifier(train);
//        accuracy = ClassifierTools.accuracy(test, c);
//
//        System.out.println("CAWPE BOSS accuracy on " + dataset + " fold 0 = " + accuracy);
////
////        c = new BOSS();
////        ((BOSS) c).setTimeLimit(TimeLimit.MINUTE, 2);
////        c.buildClassifier(train2);
////        accuracy = ClassifierTools.accuracy(test2, c);
////
////        System.out.println(((BOSS) c).numSeries);
////        System.out.println(Arrays.toString(((BOSS) c).numClassifiers));
////
////        System.out.println("Contract BOSS MV accuracy on " + dataset2 + " fold 0 = " + accuracy);
////
////        c = new BOSS();
////        ((BOSS) c).setTimeLimit(TimeLimit.MINUTE, 2);
////        c.buildClassifier(train);
////        accuracy = ClassifierTools.accuracy(test, c);
////
////        System.out.println("Contract BOSS accuracy on " + dataset + " fold 0 = " + accuracy);
////
////        c = new BOSS();
////        c.buildClassifier(train2);
////        accuracy = ClassifierTools.accuracy(test2, c);
////
////        System.out.println(((BOSS) c).numSeries);
////        System.out.println(Arrays.toString(((BOSS) c).numClassifiers));
////
////        System.out.println("BOSS MV accuracy on " + dataset2 + " fold 0 = " + accuracy);
//
    }

    /**
     * BOSS classifier to be used with known parameters, for boss with parameter search, use BOSSEnsemble.
     * 
     * Current implementation of BitWord as of 07/11/2016 only supports alphabetsize of 4, which is the expected value 
     * as defined in the paper
     * 
     * Params: wordLength, alphabetSize, windowLength, normalise?
     * 
     * @author James Large. Enhanced by original author Patrick Schaefer
     * 
     * Implementation based on the algorithm described in getTechnicalInformation()
     */
    public static class BOSSIndividual extends AbstractClassifier implements Serializable, Comparable<BOSSIndividual> {

        //all sfa words found in original buildClassifier(), no numerosity reduction/shortening applied
        protected BitWord [/*instance*/][/*windowindex*/] SFAwords;

        //histograms of words of the current wordlength with numerosity reduction applied (if selected)
        protected ArrayList<Bag> bags;

        //breakpoints to be found by MCB
        protected double[/*letterindex*/][/*breakpointsforletter*/] breakpoints;

        protected double inverseSqrtWindowSize;
        protected int windowSize;
        protected int wordLength;
        protected int alphabetSize;
        protected boolean norm;

        protected boolean numerosityReduction = true;
        protected boolean cleanAfterBuild = false;

        protected double accuracy = -1;

        protected static final long serialVersionUID = 22551L;

        public BOSSIndividual(int wordLength, int alphabetSize, int windowSize, boolean normalise) {
            this.wordLength = wordLength;
            this.alphabetSize = alphabetSize;
            this.windowSize = windowSize;
            this.inverseSqrtWindowSize = 1.0 / Math.sqrt(windowSize);
            this.norm = normalise;
        }

        /**
         * Used when shortening histograms, copies 'meta' data over, but with shorter
         * word length, actual shortening happens separately
         */
        public BOSSIndividual(BOSSIndividual boss, int wordLength) {
            this.wordLength = wordLength;

            this.windowSize = boss.windowSize;
            this.inverseSqrtWindowSize = boss.inverseSqrtWindowSize;
            this.alphabetSize = boss.alphabetSize;
            this.norm = boss.norm;
            this.numerosityReduction = boss.numerosityReduction;

            this.SFAwords = boss.SFAwords;
            this.breakpoints = boss.breakpoints;

            this.bags = new ArrayList<>(boss.bags.size());
        }

        @Override
        public int compareTo(BOSSIndividual o) {
            return Double.compare(this.accuracy, o.accuracy);
        }

        public static class Bag extends HashMap<BitWord, Integer> {
            double classVal;
            protected static final long serialVersionUID = 22552L;

            public Bag() {
                super();
            }

            public Bag(int classValue) {
                super();
                classVal = classValue;
            }

            public double getClassVal() { return classVal; }
            public void setClassVal(double classVal) { this.classVal = classVal; }       
        }

        public int getWindowSize() { return windowSize; }
        public int getWordLength() { return wordLength; }
        public int getAlphabetSize() { return alphabetSize; }
        public boolean isNorm() { return norm; }

        /**
         * @return { numIntervals(word length), alphabetSize, slidingWindowSize, normalise? } 
         */
        public int[] getParameters() {
            return new int[] { wordLength, alphabetSize, windowSize };
        }

        public void clean() {
            SFAwords = null;
        }

        protected double[][] performDFT(double[][] windows) {
            double[][] dfts = new double[windows.length][wordLength];
            for (int i = 0; i < windows.length; ++i) {
                dfts[i] = DFT(windows[i]);
            }
            return dfts;
        }

        protected double stdDev(double[] series) {
            double sum = 0.0;
            double squareSum = 0.0;
            for (int i = 0; i < windowSize; i++) {
                sum += series[i];
                squareSum += series[i]*series[i];
            }

            double mean = sum / series.length;
            double variance = squareSum / series.length - mean*mean;
            return variance > 0 ? Math.sqrt(variance) : 1.0;
        }

        protected double[] DFT(double[] series) {
            //taken from FFT.java but 
            //return just a double[] size n, { real1, imag1, ... realn/2, imagn/2 }
            //instead of Complex[] size n/2

            //only calculating first wordlength/2 coefficients (output values), 
            //and skipping first coefficient if the data is to be normalised
            int n=series.length;
            int outputLength = wordLength/2;
            int start = (norm ? 1 : 0);

            //normalize the disjoint windows and sliding windows by dividing them by their standard deviation 
            //all Fourier coefficients are divided by sqrt(windowSize)

            double normalisingFactor = inverseSqrtWindowSize / stdDev(series);

            double[] dft=new double[outputLength*2];

            for (int k = start; k < start + outputLength; k++) {  // For each output element
                float sumreal = 0;
                float sumimag = 0;
                for (int t = 0; t < n; t++) {  // For each input element
                    sumreal +=  series[t]*Math.cos(2*Math.PI * t * k / n);
                    sumimag += -series[t]*Math.sin(2*Math.PI * t * k / n);
                }
                dft[(k-start)*2]   = sumreal * normalisingFactor;
                dft[(k-start)*2+1] = sumimag * normalisingFactor;
            }
            return dft;
        }

        private double[] DFTunnormed(double[] series) {
            //taken from FFT.java but 
            //return just a double[] size n, { real1, imag1, ... realn/2, imagn/2 }
            //instead of Complex[] size n/2

            //only calculating first wordlength/2 coefficients (output values), 
            //and skipping first coefficient if the data is to be normalised
            int n=series.length;
            int outputLength = wordLength/2;
            int start = (norm ? 1 : 0);

            double[] dft = new double[outputLength*2];
            double twoPi = 2*Math.PI / n;

            for (int k = start; k < start + outputLength; k++) {  // For each output element
                float sumreal = 0;
                float sumimag = 0;
                for (int t = 0; t < n; t++) {  // For each input element
                    sumreal +=  series[t]*Math.cos(twoPi * t * k);
                    sumimag += -series[t]*Math.sin(twoPi * t * k);
                }
                dft[(k-start)*2]   = sumreal;
                dft[(k-start)*2+1] = sumimag;
            }
            return dft;
        }

        private double[] normalizeDFT(double[] dft, double std) {
            double normalisingFactor = (std > 0? 1.0 / std : 1.0) * inverseSqrtWindowSize;
            for (int i = 0; i < dft.length; i++)
                dft[i] *= normalisingFactor;

            return dft;
        }

        private double[][] performMFT(double[] series) {
            // ignore DC value?
            int startOffset = norm ? 2 : 0;
            int l = wordLength;
            l = l + l % 2; // make it even
            double[] phis = new double[l];
            for (int u = 0; u < phis.length; u += 2) {
                double uHalve = -(u + startOffset) / 2;
                phis[u] = realephi(uHalve, windowSize);
                phis[u + 1] = complexephi(uHalve, windowSize);
            }

            // means and stddev for each sliding window
            int end = Math.max(1, series.length - windowSize + 1);
            double[] means = new double[end];
            double[] stds = new double[end];
            calcIncrementalMeanStddev(windowSize, series, means, stds);
            // holds the DFT of each sliding window
            double[][] transformed = new double[end][];
            double[] mftData = null;

            for (int t = 0; t < end; t++) {
                // use the MFT
                if (t > 0) {
                    for (int k = 0; k < l; k += 2) {
                        double real1 = (mftData[k] + series[t + windowSize - 1] - series[t - 1]);
                        double imag1 = (mftData[k + 1]);
                        double real = complexMulReal(real1, imag1, phis[k], phis[k + 1]);
                        double imag = complexMulImag(real1, imag1, phis[k], phis[k + 1]);
                        mftData[k] = real;
                        mftData[k + 1] = imag;
                    }
                } // use the DFT for the first offset
                else {
                    mftData = Arrays.copyOf(series, windowSize);
                    mftData = DFTunnormed(mftData);
                }
                // normalization for lower bounding
                transformed[t] = normalizeDFT(Arrays.copyOf(mftData, l), stds[t]);
            }
            return transformed;
        }

        private void calcIncrementalMeanStddev(int windowLength, double[] series, double[] means, double[] stds) {
            double sum = 0;
            double squareSum = 0;
            // it is faster to multiply than to divide
            double rWindowLength = 1.0 / (double) windowLength;
            double[] tsData = series;
            for (int ww = 0; ww < windowLength; ww++) {
                sum += tsData[ww];
                squareSum += tsData[ww] * tsData[ww];
            }
            means[0] = sum * rWindowLength;
            double buf = squareSum * rWindowLength - means[0] * means[0];
            stds[0] = buf > 0 ? Math.sqrt(buf) : 0;
            for (int w = 1, end = tsData.length - windowLength + 1; w < end; w++) {
                sum += tsData[w + windowLength - 1] - tsData[w - 1];
                means[w] = sum * rWindowLength;
                squareSum += tsData[w + windowLength - 1] * tsData[w + windowLength - 1] - tsData[w - 1] * tsData[w - 1];
                buf = squareSum * rWindowLength - means[w] * means[w];
                stds[w] = buf > 0 ? Math.sqrt(buf) : 0;
            }
        }

        private static double complexMulReal(double r1, double im1, double r2, double im2) {
            return r1 * r2 - im1 * im2;
        }

        private static double complexMulImag(double r1, double im1, double r2, double im2) {
            return r1 * im2 + r2 * im1;
        }

        private static double realephi(double u, double M) {
            return Math.cos(2 * Math.PI * u / M);
        }

        private static double complexephi(double u, double M) {
            return -Math.sin(2 * Math.PI * u / M);
        }

        protected double[][] disjointWindows(double [] data) {
            int amount = (int)Math.ceil(data.length/(double)windowSize);
            double[][] subSequences = new double[amount][windowSize];

            for (int win = 0; win < amount; ++win) { 
                int offset = Math.min(win*windowSize, data.length-windowSize);

                //copy the elements windowStart to windowStart+windowSize from data into 
                //the subsequence matrix at position windowStart
                System.arraycopy(data,offset,subSequences[win],0,windowSize);
            }

            return subSequences;
        }

        protected double[][] MCB(Instances data) {
            double[][][] dfts = new double[data.numInstances()][][];

            int sample = 0;
            for (Instance inst : data)
                dfts[sample++] = performDFT(disjointWindows(toArrayNoClass(inst))); //approximation

            int numInsts = dfts.length;
            int numWindowsPerInst = dfts[0].length;
            int totalNumWindows = numInsts*numWindowsPerInst;

            breakpoints = new double[wordLength][alphabetSize]; 

            for (int letter = 0; letter < wordLength; ++letter) { //for each dft coeff

                //extract this column from all windows in all instances
                double[] column = new double[totalNumWindows];
                for (int inst = 0; inst < numInsts; ++inst)
                    for (int window = 0; window < numWindowsPerInst; ++window) {
                        //rounding dft coefficients to reduce noise
                        column[(inst * numWindowsPerInst) + window] = Math.round(dfts[inst][window][letter]*100.0)/100.0;   
                    }

                //sort, and run through to find breakpoints for equi-depth bins
                Arrays.sort(column);

                double binIndex = 0;
                double targetBinDepth = (double)totalNumWindows / (double)alphabetSize; 

                for (int bp = 0; bp < alphabetSize-1; ++bp) {
                    binIndex += targetBinDepth;
                    breakpoints[letter][bp] = column[(int)binIndex];
                }

                breakpoints[letter][alphabetSize-1] = Double.MAX_VALUE; //last one can always = infinity
            }

            return breakpoints;
        }

        /**
         * Builds a brand new boss bag from the passed fourier transformed data, rather than from
         * looking up existing transforms from earlier builds (i.e. SFAWords). 
         * 
         * to be used e.g to transform new test instances
         */
        protected Bag createBagSingle(double[][] dfts) {
            Bag bag = new Bag();
            BitWord lastWord = new BitWord();

            for (double[] d : dfts) {
                BitWord word = createWord(d);
                //add to bag, unless num reduction applies
                if (numerosityReduction && word.equals(lastWord))
                    continue;

                Integer val = bag.get(word);
                if (val == null)
                    val = 0;
                bag.put(word, ++val);   

                lastWord = word;
            }

            return bag;
        }

        protected BitWord createWord(double[] dft) {
            BitWord word = new BitWord(wordLength);
            for (int l = 0; l < wordLength; ++l) //for each letter
                for (int bp = 0; bp < alphabetSize; ++bp) //run through breakpoints until right one found
                    if (dft[l] <= breakpoints[l][bp]) {
                        word.push(bp); //add corresponding letter to word
                        break;
                    }

            return word;
        }

        /**
         * @return data of passed instance in a double array with the class value removed if present
         */
        protected static double[] toArrayNoClass(Instance inst) {
            int length = inst.numAttributes();
            if (inst.classIndex() >= 0)
                --length;

            double[] data = new double[length];

            for (int i=0, j=0; i < inst.numAttributes(); ++i)
                if (inst.classIndex() != i)
                    data[j++] = inst.value(i);

            return data;
        }

        /**
         * @return BOSSTransform-ed bag, built using current parameters
         */
        public Bag BOSSTransform(Instance inst) {
            double[][] mfts = performMFT(toArrayNoClass(inst)); //approximation     
            Bag bag = createBagSingle(mfts); //discretisation/bagging
            bag.setClassVal(inst.classValue());

            return bag;
        }

        /**
         * Shortens all bags in this BOSS instance (histograms) to the newWordLength, if wordlengths
         * are same, instance is UNCHANGED
         *
         * @param newWordLength wordLength to shorten it to
         * @return new boss classifier with newWordLength, or passed in classifier if wordlengths are same
         */
        public BOSSIndividual buildShortenedBags(int newWordLength) throws Exception {
            if (newWordLength == wordLength) //case of first iteration of word length search in ensemble
                return this;
            if (newWordLength > wordLength)
                throw new Exception("Cannot incrementally INCREASE word length, current:"+wordLength+", requested:"+newWordLength);
            if (newWordLength < 2)
                throw new Exception("Invalid wordlength requested, current:"+wordLength+", requested:"+newWordLength);

            BOSSIndividual newBoss = new BOSSIndividual(this, newWordLength);

            //build hists with new word length from SFA words, and copy over the class values of original insts
            for (int i = 0; i < bags.size(); ++i) {
                Bag newBag = createBagFromWords(newWordLength, SFAwords[i]);
                newBag.setClassVal(bags.get(i).getClassVal());
                newBoss.bags.add(newBag);
            }

            return newBoss;
        }

        /**
         * Builds a bag from the set of words for a pre-transformed series of a given wordlength.
         */
        protected Bag createBagFromWords(int thisWordLength, BitWord[] words) {
            Bag bag = new Bag();
            BitWord lastWord = new BitWord();

            for (BitWord w : words) {
                BitWord word = new BitWord(w);
                if (wordLength != thisWordLength)
                    word.shorten(16-thisWordLength); 
                    //TODO hack, word.length=16=maxwordlength, wordLength of 'this' BOSS instance unreliable, length of SFAwords = maxlength

                //add to bag, unless num reduction applies
                if (numerosityReduction && word.equals(lastWord))
                    continue;

                Integer val = bag.get(word);
                if (val == null)
                    val = 0;
                bag.put(word, ++val);   

                lastWord = word;
            }

            return bag;
        }

        protected BitWord[] createSFAwords(Instance inst) {
            double[][] dfts = performMFT(toArrayNoClass(inst)); //approximation     
            BitWord[] words = new BitWord[dfts.length];
            for (int window = 0; window < dfts.length; ++window) 
                words[window] = createWord(dfts[window]);//discretisation

            return words;
        }

        @Override
        public void buildClassifier(Instances data) throws Exception {
            if (data.classIndex() != data.numAttributes()-1)
                throw new Exception("BOSS_BuildClassifier: Class attribute not set as last attribute in dataset");

            breakpoints = MCB(data); //breakpoints to be used for making sfa words for train AND test data
            SFAwords = new BitWord[data.numInstances()][];
            bags = new ArrayList<>(data.numInstances());

            //1NN BOSS distance
            for (int inst = 0; inst < data.numInstances(); ++inst) {
                SFAwords[inst] = createSFAwords(data.get(inst));

                Bag bag = createBagFromWords(wordLength, SFAwords[inst]);
                bag.setClassVal(data.get(inst).classValue());
                bags.add(bag);
            }

            if (cleanAfterBuild) {
                clean();
            }
        }

        /**
         * Computes BOSS distance between two bags d(test, train), is NON-SYMETRIC operation, ie d(a,b) != d(b,a).
         * 
         * Quits early if the dist-so-far is greater than bestDist (assumed dist is still the squared distance), and returns Double.MAX_VALUE
         * 
         * @return distance FROM instA TO instB, or Double.MAX_VALUE if it would be greater than bestDist
         */
        public double BOSSdistance(Bag instA, Bag instB, double bestDist) {
            double dist = 0.0;

            //find dist only from values in instA
            for (Entry<BitWord, Integer> entry : instA.entrySet()) {
                Integer valA = entry.getValue();
                Integer valB = instB.get(entry.getKey());
                if (valB == null)
                    valB = 0;
                dist += (valA-valB)*(valA-valB);

                if (dist > bestDist)
                    return Double.MAX_VALUE;
            }

            return dist;
        }

        @Override
        public double classifyInstance(Instance instance) throws Exception {
            Bag testBag = BOSSTransform(instance);

            double bestDist = Double.MAX_VALUE;

            //1NN BOSS distance
            double nn = -1.0;

            //find dist FROM testBag TO all trainBags
            for (int i = 0; i < bags.size(); ++i) {
                double dist = BOSSdistance(testBag, bags.get(i), bestDist);

                if (dist < bestDist) {
                    bestDist = dist;
                    nn = bags.get(i).getClassVal();
                }
            }

            return nn;
        }

        /**
         * Used within BOSSEnsemble as part of a leave-one-out crossvalidation, to skip having to rebuild 
         * the classifier every time (since the n histograms would be identical each time anyway), therefore this classifies 
         * the instance at the index passed while ignoring its own corresponding histogram 
         * 
         * @param testIndex index of instance to classify
         * @return classification
         */
        public double classifyInstance(int testIndex) throws Exception {
            double bestDist = Double.MAX_VALUE;
            Bag testBag = bags.get(testIndex);

            //1NN BOSS distance
            double nn = -1.0;

            for (int i = 0; i < bags.size(); ++i) {
                if (i == testIndex) //skip 'this' one, leave-one-out
                    continue;

                double dist = BOSSdistance(testBag, bags.get(i), bestDist);

                if (dist < bestDist) {
                    bestDist = dist;
                    nn = bags.get(i).getClassVal();
                }
            }

            return nn;
        }
    }
}
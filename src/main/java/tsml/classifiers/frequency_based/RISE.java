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
package tsml.classifiers.frequency_based;

import evaluation.evaluators.SingleSampleEvaluator;
import evaluation.storage.ClassifierResults;
import evaluation.tuning.ParameterSpace;
import experiments.data.DatasetLists;
import fileIO.FullAccessOutFile;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.Tuneable;
import tsml.filters.*;
import utilities.ClassifierTools;
import utilities.multivariate_tools.MultivariateInstanceTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomTree;
import weka.core.*;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import tsml.classifiers.Checkpointable;
import tsml.classifiers.TrainTimeContractable;

import static experiments.data.DatasetLoading.loadDataNullable;

/**
 <!-- globalinfo-start -->
 * Variation of Random Interval Spectral Ensemble [lines2018time].
 *
 * This implementation extends the original to include:
 * down sampling
 * stabilisation (constraining interval length to within some distance of previous length)
 * check pointing
 * contracting
 *
 * Overview: Input n series length m
 * for each tree
 *      sample interval of random size
 *      transform interval into ACF and PS features
 *      build tree on concatenated features
 * ensemble the trees with majority vote
 <!-- globalinfo-end -->
 <!-- technical-bibtex-start -->
 * Bibtex
 * <pre>
 *   @article{lines2018time,
 *   title={Time series classification with HIVE-COTE: The hierarchical vote collective of transformation-based ensembles},
 *   author={Lines, Jason and Taylor, Sarah and Bagnall, Anthony},
 *   journal={ACM Transactions on Knowledge Discovery from Data (TKDD)},
 *   volume={12},
 *   number={5},
 *   pages={52},
 *   year={2018},
 *   publisher={ACM}
 *   }
 * </pre>
 <!-- technical-bibtex-end -->
 <!-- options-start -->
 <!-- options-end -->
 * @author Michael Flynn and Tony Bagnall
 * @date 19/02/19
 * updated 4/3/20 to conform to tsml standards
 **/

public class RISE extends EnhancedAbstractClassifier implements TrainTimeContractable, TechnicalInformationHandler, Checkpointable, Tuneable {
//This needs better commenting of what these variables mean
    private int maxIntervalLength = 0;
    private int minIntervalLength = 16;
    private int numClassifiers = 500;
    private int treeCount = 0;
    private int minNumTrees = 0;
    private boolean downSample = false;
    private boolean loadedFromFile = false;
    private int stabilise = 0;
    private final int DEFAULT_MAXLAG = 100;
    private final int DEFAULT_MINLAG = 1;
    private double perForBag = 0.5;

    private Timer timer = null;

    private Classifier classifier = new RandomTree();
    private ArrayList<Classifier> baseClassifiers = null;
    private ArrayList<int[]> intervalsInfo = null;
    private ArrayList<ArrayList<Integer>> intervalsAttIndexes = null;
    private ArrayList<Integer> rawIntervalIndexes = null;
    private PowerSpectrum PS;
    private TransformType transformType = TransformType.ACF_PS;
    private String serialisePath = null;
    private Instances data = null;

    //Updated work
    private ArrayList<int[]> startEndPoints = null;

    /**
     * Constructor
     * @param seed
     */
    public RISE(long seed){
/**
 * to do: make it so it can estimate own performance
  */
        super(CAN_ESTIMATE_OWN_PERFORMANCE);
        super.setSeed((int)seed);
        timer = new Timer();
    }

    public RISE(){
// DO THIS INSTEAD
//      this(0);
        super(CAN_ESTIMATE_OWN_PERFORMANCE);
        this.seed = 0;
        super.setSeed((int)seed);
        timer = new Timer();
/**
 * Why is this not done in the seeded constructor?
  */
        this.setTransformType(TransformType.ACF_FFT);
    }

    public enum TransformType {ACF, FACF, PS, FFT, FACF_FFT, ACF_FFT, ACF_PS, ACF_PS_AR, MFCC}

    /**
     * Function used to reset internal state of classifier.
     * Is called at beginning of buildClassifier.
 Can subsequently call buildClassifier multiple times per instance of RISE.
     */
    private void initialise(){
        timer.reset();
        baseClassifiers = new ArrayList<>();
        intervalsInfo = new ArrayList<>();
        intervalsAttIndexes = new ArrayList<>();
        rawIntervalIndexes = new ArrayList<>();
        startEndPoints = new ArrayList<>();
        PS = new PowerSpectrum();
        treeCount = 0;
    }

    /**
     * Sets number of trees.
     * @param numClassifiers
     */
    public void setNumClassifiers(int numClassifiers){
        this.numClassifiers = numClassifiers;
    }

    /**
     * Sets minimum number of trees RISE will build if contracted.
     * @param minNumTrees
     */
    public void setMinNumTrees(int minNumTrees){
        this.minNumTrees = minNumTrees;
    }

    /**
     * Boolean to set downSample.
     * If true down sample rate is randomly selected per interval.
     * @param bool
     */
    public void setDownSample(boolean bool){
        this.downSample = bool;
    }

    /**
     * Parameter to control width of interval space with prior interval length centered.
     * e.g. priorIntervalLength = 53
     *      width = 7
     *      possibleWidths = 50 < x < 56 (inclusive)
     * Has the effect of constraining the space around the previous interval length, contributing to a more robust
     * timing model via preventing leveraging in large problems.
     * @param width
     */
    public void setStabilise(int width){
        this.stabilise = width;
    }

    /**
     * Location of folder in which to save timing model information.
     * @param modelOutPath
     */
    public void setModelOutPath(String modelOutPath){
        timer.modelOutPath = modelOutPath;
    }

    /**
     * Default transform combined ACF+PS
     * @param transformType
     */
    public void setTransformType(TransformType transformType){
        this.transformType = transformType;
    }

    /**
     * Pass in instance of {@code weka.classifiers.trees} to replace default base classifier.
     * @param classifier
     */
    public void setBaseClassifier(Classifier classifier){
        this.classifier = classifier;
    }

    public void setPercentageOfContractForBagging(double x){
        perForBag = x;
    }

    /**
     * RISE will attempt to load serialisation file on method call using the seed set on instantiation as file
 identifier.
     * If successful this object is returned to state in which it was at creation of serialisation file.
     * @param serialisePath Path to folder in which to save serialisation files.
     */
    @Override //Checkpointable
    public boolean setSavePath(String serialisePath) {
        boolean validPath=Checkpointable.super.setSavePath(serialisePath);
        if(validPath){
            this.serialisePath = serialisePath;
        }
        return validPath;
    }

/*
    @Override
    public void setSavePath(String serializePath){
        this.serialisePath = serializePath;
        System.out.println("Attempting to load from file location: "
                + serialisePath
                + "\\SERIALISE_cRISE_"
                + seed
                + ".txt");
        cRISE temp = readSerialise(seed);
        copyFromSerObject(temp);
    }
*/
    public int getMaxLag(Instances instances){
        int maxLag = (instances.numAttributes()-1)/4;
        if(DEFAULT_MAXLAG < maxLag)
            maxLag = DEFAULT_MAXLAG;
        /*if(maxLag < DEFAULT_MINLAG)
            maxLag = (instances.numAttributes()-1);*/
        return maxLag;
    }

    /**
     * Method controlling interval length, interval start position and down sample factor (if set).
     * Takes into account stabilisation parameter if set.
     * @param maxIntervalLength maximum length interval can be in order to adhere to minimum number of trees and contract constraints.
     * @param instanceLength
     * @return int[] of size three:
     *      int[0] = rawIntervalLength
     *      int[1] = startIndex
     *      int[2] = downSampleFactor
     */
    private int[] selectIntervalAttributes(int maxIntervalLength, int instanceLength){

        //rawIntervalLength[0], startIndex[1], downSampleFactor[2];
        int[] intervalInfo = new int[3];

        //Produce powers of 2 ArrayList for interval selection.
        ArrayList<Integer> powersOf2 = new ArrayList<>();
        for (int j = maxIntervalLength; j >= 1; j--) {
            // If j is a power of 2
            if ((j & (j - 1)) == 0){
                powersOf2.add(j);
            }
        }

        Collections.reverse(powersOf2);
        int index = 0;

        //If stabilise is set.
        if(stabilise > 0 && !rawIntervalIndexes.isEmpty()){
            //Check stabilise is valid value.
            if(stabilise > powersOf2.size()-1){
                stabilise = powersOf2.size()-1;
                while(stabilise % 2 == 0){
                    stabilise --;
                }
            }else if(stabilise < 2){
                stabilise = 2;
                while(stabilise % 2 == 0){
                    stabilise ++;
                }
            }else{
                while(stabilise % 2 == 0){
                    stabilise ++;
                }
            }

            //Select random value between 0 - (stabilise - 1)
            //Map value onto valid interval length based on previous length, correcting for occasions in which previous
            //length = 0 | length = maxLength.
            int option = rand.nextInt(stabilise - 1);
            if(rawIntervalIndexes.get(rawIntervalIndexes.size()-1) - ((stabilise - 1)/2) <= 2){
                index = option + 2;
            }
            if (rawIntervalIndexes.get(rawIntervalIndexes.size()-1) - ((stabilise - 1)/2) > 2 && rawIntervalIndexes.get(rawIntervalIndexes.size()-1) + ((stabilise - 1)/2) < powersOf2.size() - 1) {
                option = option - ((stabilise - 1)/2);
                index = rawIntervalIndexes.get(rawIntervalIndexes.size()-1) + option;
            }
            if(rawIntervalIndexes.get(rawIntervalIndexes.size()-1) + ((stabilise - 1)/2) >= powersOf2.size() - 1) {
                index = (powersOf2.size() - 1) - option;
            }
        }else{
            //If stabilise is not set.
            //Select a new interval length at random (Selects in linear space and maps onto closest power of two).
            int temp = rand.nextInt(powersOf2.get(powersOf2.size() - 1)) + 1;
            while((temp & (temp - 1)) != 0)
                temp++;

            for (int i = 0; i < powersOf2.size() && temp != powersOf2.get(i); i++) {
                index = i;
            }
            index++;
        }

        //If this tree is one of first four trees use tree number as powersOf2 index. Establishes robust foundation for
        //timing model. However, logic should be refactored to check this before executing prior code.
        try{
            if(treeCount < 4){
                index = (treeCount + 2) < powersOf2.size()-1 ? (treeCount + 2) : powersOf2.size()-1;
            }
            if(treeCount == 4){
                index = powersOf2.size()-1;
            }
            intervalInfo[0] = powersOf2.get(index);
        }catch(Exception e){
            System.out.println(e);
        }

        //Select random start index to take interval from.
        if ((instanceLength - intervalInfo[0]) != 0 ) {
            intervalInfo[1] = rand.nextInt(instanceLength - intervalInfo[0]);
        }else{
            intervalInfo[1] = 0;
        }

        //Select down sample factor such that it is a smaller or equal power of 2 whilst ensuring resulting interval
        //length is also a power of 2.
        //e.g. if length is 8 down sample factor can be 1, 2, 4 or 8. Results in step lengths of, 8(8/1), 4(8/2), 2(8/4) or 1(8/8)
        //and total interval lengths of 1, 2, 4 or 8.
        if (downSample) {
            intervalInfo[2] = powersOf2.get(rand.nextInt(index) + 1);
        }else{
            intervalInfo[2] = intervalInfo[0];
        }

        this.intervalsInfo.add(intervalInfo);
        this.rawIntervalIndexes.add(index);
        return intervalInfo;
    }

    /**
     * Produces interval instances based on logic executed in {@code selectIntervalAttributes}.
     * @param maxIntervalLength
     * @param trainingData
     * @return new training set transformed such that instances are now an interval of original instances.
     */
    private Instances produceIntervalInstances(int maxIntervalLength, Instances trainingData){

        Instances intervalInstances;
        ArrayList<Attribute>attributes = new ArrayList<>();
        int[] intervalInfo = selectIntervalAttributes(maxIntervalLength, trainingData.numAttributes() - 1);
        ArrayList<Integer> intervalAttIndexes = new ArrayList<>();

        for (int i = intervalInfo[1]; i < (intervalInfo[1] + intervalInfo[0]); i += (intervalInfo[0] / intervalInfo[2])) {
            attributes.add(trainingData.attribute(i));
            intervalAttIndexes.add(i);
        }

        intervalsAttIndexes.add(intervalAttIndexes);
        attributes.add(trainingData.attribute(trainingData.numAttributes()-1));
        intervalInstances = new Instances(trainingData.relationName(), attributes, trainingData.size());
        double[] intervalInstanceValues = new double[intervalInfo[2] + 1];

        for (int i = 0; i < trainingData.size(); i++) {

            for (int j = 0; j < intervalInfo[2]; j++) {
                intervalInstanceValues[j] = trainingData.get(i).value(intervalAttIndexes.get(j));
            }

            DenseInstance intervalInstance = new DenseInstance(intervalInstanceValues.length);
            intervalInstance.replaceMissingValues(intervalInstanceValues);
            intervalInstance.setValue(intervalInstanceValues.length-1, trainingData.get(i).classValue());
            intervalInstances.add(intervalInstance);
        }

        intervalInstances.setClassIndex(intervalInstances.numAttributes() - 1);

        return intervalInstances;
    }

    /**
     * Transforms test instance into interval instance based on classifierNumber.
     * @param testInstance
     * @param classifierNum
     * @return new test instance transformed such that it is now an interval of original instance.
     */
    private Instance produceIntervalInstance(Instance testInstance, int classifierNum){
        double[] instanceValues = new double[intervalsAttIndexes.get(classifierNum).size() + 1];
        ArrayList<Attribute>attributes = new ArrayList<>();

        for (int i = 0; i < intervalsAttIndexes.get(classifierNum).size(); i++) {
            attributes.add(testInstance.attribute(i));
            instanceValues[i] = testInstance.value(intervalsAttIndexes.get(classifierNum).get(i));
        }

        Instances testInstances = new Instances("relationName", attributes, 1);
        instanceValues[instanceValues.length - 1] = testInstance.value(testInstance.numAttributes() - 1);
        Instance temp = new DenseInstance(instanceValues.length);
        temp.replaceMissingValues(instanceValues);
        testInstances.add(temp);
        testInstances.setClassIndex(testInstances.numAttributes() - 1);

        return testInstances.firstInstance();
    }

    private Instances produceIntervalInstanceUpdate(Instance testInstance, int classifierNum){
        Instances intervalInstances = null;
        ArrayList<Attribute>attributes = new ArrayList<>();
        //int nearestPowerOfTwo = (int)FFT.MathsPower2.roundPow2((float) startEndPoints.get(classifierNum)[1] - startEndPoints.get(classifierNum)[0]);
        int nearestPowerOfTwo = startEndPoints.get(classifierNum)[1] - startEndPoints.get(classifierNum)[0];

        for (int i = 0; i < nearestPowerOfTwo; i ++) {
            Attribute att = i + startEndPoints.get(classifierNum)[0] < testInstance.numAttributes() - 1 ? testInstance.attribute(i + startEndPoints.get(classifierNum)[0]) : new Attribute("att"+ (i + 1 + startEndPoints.get(classifierNum)[0]));
            //Attribute att = i + startEndPoints.get(classifierNum)[0] < (startEndPoints.get(classifierNum)[1] - startEndPoints.get(classifierNum)[0]) ? testInstance.attribute(i + startEndPoints.get(classifierNum)[0]) : new Attribute("att"+ (i + 1 + startEndPoints.get(classifierNum)[0]));
            attributes.add(att);
        }

        attributes.add(testInstance.attribute(testInstance.numAttributes()-1));
        intervalInstances = new Instances(testInstance.dataset().relationName(), attributes, 1);
        double[] intervalInstanceValues = new double[nearestPowerOfTwo + 1];

        for (int j = 0; j < nearestPowerOfTwo; j++) {
            double value = j + startEndPoints.get(classifierNum)[0] < testInstance.numAttributes() - 1 ? testInstance.value(j + startEndPoints.get(classifierNum)[0]) : 0.0;
            //double value = j + startEndPoints.get(classifierNum)[0] < (startEndPoints.get(classifierNum)[1] - startEndPoints.get(classifierNum)[0]) ? testInstance.value(j + startEndPoints.get(classifierNum)[0]) : 0.0;
            intervalInstanceValues[j] = value;
        }

        DenseInstance intervalInstance = new DenseInstance(intervalInstanceValues.length);
        intervalInstance.replaceMissingValues(intervalInstanceValues);
        intervalInstance.setValue(intervalInstanceValues.length-1, testInstance.classValue());
        intervalInstances.add(intervalInstance);

        intervalInstances.setClassIndex(intervalInstances.numAttributes() - 1);

        return intervalInstances;
    }

    /**
     * Transforms instances into either PS ACF or concatenation based on {@code setTransformType}
     * @param instances
     * @return transformed instances.
     */
    private Instances transformInstances(Instances instances, TransformType transformType){
        Instances temp = null;

        switch(transformType){
            case ACF:
                ACF acf = new ACF();
                //acf.setMaxLag(getMaxLag(instances));
                acf.setNormalized(false);
                try {
                    temp = acf.process(instances);
                } catch (Exception e) {
                    System.out.println(" Exception in Combo="+e+" max lag =" + (instances.get(0).numAttributes()-1/4));
                }
                break;
            case PS:
                try {
                    PS.useFFT();
                    temp = PS.process(instances);
                } catch (Exception ex) {
                    System.out.println("FFT failed (could be build or classify) \n" + ex);
                }
                break;
            case FFT:
                Fast_FFT Fast_FFT = new Fast_FFT();
                try {
                    //int nfft = (int)FFT.MathsPower2.roundPow2(instances.numAttributes()-1) < instances.numAttributes()-1 ? ((int)FFT.MathsPower2.roundPow2(instances.numAttributes()-1) * 2) : ((int)FFT.MathsPower2.roundPow2(instances.numAttributes()-1));
                    int nfft = (int)FFT.MathsPower2.roundPow2(instances.numAttributes()-1) * 2;
                    Fast_FFT.setNFFT(nfft);
                    temp = Fast_FFT.process(instances);
                } catch (Exception e) {
                    e.printStackTrace();
                }
                break;
            case MFCC:
                MFCC MFCC= new MFCC();
                try {
                    Instances temptemp;
                    temptemp = MFCC.process(instances);
                    temp = MFCC.determineOutputFormatForFirstChannel(instances);
                    Instance[] temptemptemp = MultivariateInstanceTools.splitMultivariateInstanceWithClassVal(temptemp.get(0));
                    for (int i = 0; i < instances.size(); i++) {
                        temp.add(temptemptemp[i]);
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
                break;

            case ACF_PS:
                temp = transformInstances(instances, TransformType.PS);
                temp.setClassIndex(-1);
                temp.deleteAttributeAt(temp.numAttributes()-1);
                temp = Instances.mergeInstances(temp, transformInstances(instances, TransformType.ACF));
                temp.setClassIndex(temp.numAttributes()-1);
                break;
            case ACF_FFT:
                temp = transformInstances(instances, TransformType.FFT);
                temp.setClassIndex(-1);
                temp.deleteAttributeAt(temp.numAttributes()-1);
                temp = Instances.mergeInstances(temp, transformInstances(instances, TransformType.ACF));
                temp.setClassIndex(temp.numAttributes()-1);
                break;
            case FACF_FFT:
                temp = transformInstances(instances, TransformType.FACF);
                try{
                    temp.setClassIndex(-1);
                }catch(Exception e){
                    temp.setClassIndex(-1);
                }
                temp.setClassIndex(-1);
                temp.deleteAttributeAt(temp.numAttributes()-1);
                temp = Instances.mergeInstances(temp, transformInstances(instances, TransformType.FFT));
                temp.setClassIndex(temp.numAttributes()-1);
                break;
            case ACF_PS_AR:
                temp = transformInstances(instances, TransformType.ACF_PS);
                temp.setClassIndex(-1);
                temp.deleteAttributeAt(temp.numAttributes()-1);

                ARMA arma=new ARMA();
                arma.setMaxLag(getMaxLag(instances));
                arma.setUseAIC(false);
                Instances arData = null;
                try {
                    arData = arma.process(instances);
                } catch (Exception e) {
                    e.printStackTrace();
                }
                temp = Instances.mergeInstances(temp, arData);
                temp.setClassIndex(temp.numAttributes()-1);
                break;
        }
        return temp;
    }

    private void saveToFile(long seed){
        try{
            System.out.println("Serialising classifier.");
            File file = new File(serialisePath
                    + (serialisePath.isEmpty()? "SERIALISE_cRISE_" : "\\SERIALISE_cRISE_")
                    + seed
                    + ".txt");
            file.setWritable(true, false);
            file.setExecutable(true, false);
            file.setReadable(true, false);
            FileOutputStream f = new FileOutputStream(file);
            ObjectOutputStream o = new ObjectOutputStream(f);
            this.timer.forestElapsedTime = System.nanoTime() - this.timer.forestStartTime;
            o.writeObject(this);
            o.close();
            f.close();
            System.out.println("Serialisation completed: " + treeCount + " trees");
        } catch (IOException ex) {
            System.out.println("Serialisation failed: " + ex);
        }
    }

    private RISE readSerialise(long seed){
        ObjectInputStream oi = null;
        RISE temp = null;
        try {
            FileInputStream fi = new FileInputStream(new File(
                    serialisePath
                            + (serialisePath.isEmpty()? "SERIALISE_cRISE_" : "\\SERIALISE_cRISE_")
                            + seed
                            + ".txt"));
            oi = new ObjectInputStream(fi);
            temp = (RISE)oi.readObject();
            oi.close();
            fi.close();
            System.out.println("File load successful: " + ((RISE)temp).treeCount + " trees.");
        } catch (IOException | ClassNotFoundException ex) {
            System.out.println("File load: failed.");
        }
        return temp;
    }

    @Override
    public void copyFromSerObject(Object temp){

        try{
            this.baseClassifiers = ((RISE)temp).baseClassifiers;
            this.classifier = ((RISE)temp).classifier;
            this.data = ((RISE)temp).data;
            this.downSample = ((RISE)temp).downSample;
            this.PS = ((RISE)temp).PS;
            this.intervalsAttIndexes = ((RISE)temp).intervalsAttIndexes;
            this.intervalsInfo = ((RISE)temp).intervalsInfo;
            this.maxIntervalLength = ((RISE)temp).maxIntervalLength;
            this.minIntervalLength = ((RISE)temp).minIntervalLength;
            this.numClassifiers = ((RISE)temp).numClassifiers;
            this.rand = ((RISE)temp).rand;
            this.rawIntervalIndexes = ((RISE)temp).rawIntervalIndexes;
            this.serialisePath = ((RISE)temp).serialisePath;
            this.stabilise = ((RISE)temp).stabilise;
            this.timer = ((RISE)temp).timer;
            this.transformType = ((RISE)temp).transformType;
            this.treeCount = ((RISE)temp).treeCount;
            this.startEndPoints = ((RISE)temp).startEndPoints;
            this.loadedFromFile = true;
            System.out.println("Varible assignment: successful.");
        }catch(Exception ex){
            System.out.println("Varible assignment: unsuccessful.");
        }

    }

    /**
     * Method to maintain timing, takes into consideration that object may have been read from file and therefore be
     * mid way through a contract.
     * @return
     */
    private long getTime(){
        long time = 0;
        if(loadedFromFile){
            time = timer.forestElapsedTime;
        }else{
            time = 0;
        }
        return time;
    }

    /**
     * Build classifier
     * @param trainingData whole training set.
     * @throws Exception
     */
    @Override
    public void buildClassifier(Instances trainingData) throws Exception {

        if(serialisePath != null){
            RISE temp = this.readSerialise(seed);
            if(temp != null) {
                this.copyFromSerObject(temp);
                this.loadedFromFile = true;
            }
        }

        //If not loaded from file e.g. Starting fresh experiment.
        if (!loadedFromFile) {
            //Just used for getParameters.
            data = trainingData;
            //(re)Initialise all variables to account for multiple calls of buildClassifier.
            initialise();

            //Check min & max interval lengths are valid.
            if(maxIntervalLength > trainingData.numAttributes()-1 || maxIntervalLength <= 0){
                maxIntervalLength = trainingData.numAttributes()-1;
            }
            if(minIntervalLength >= trainingData.numAttributes()-1 || minIntervalLength <= 0){
                minIntervalLength = (trainingData.numAttributes()-1)/2;
            }

        }

        //Start forest timer.
        timer.forestStartTime = System.nanoTime();

        if (getEstimateOwnPerformance()) {
            findTrainAcc(data);
            initialise();
            timer.reset();
            this.setTrainTimeLimit(TimeUnit.NANOSECONDS, (timer.forestTimeLimit*2) - timer.forestElapsedTime);
        }

        for (; treeCount < numClassifiers && (System.nanoTime() - timer.forestStartTime) < (timer.forestTimeLimit - getTime()); treeCount++) {

            //Start tree timer.
            timer.treeStartTime = System.nanoTime();

            //Compute maximum interval length given time remaining.
            timer.buildModel();
            maxIntervalLength = (int)timer.getFeatureSpace((timer.forestTimeLimit) - (System.nanoTime() - (timer.forestStartTime - getTime())));


            //Produce intervalInstances from trainingData using interval attributes.
            Instances intervalInstances;
            //intervalInstances = produceIntervalInstances(maxIntervalLength, trainingData);
            intervalInstances = produceIntervalInstancesUpdate(maxIntervalLength, trainingData);

            //Transform instances.
            if (transformType != null) {
                intervalInstances = transformInstances(intervalInstances, transformType);
            }

            //Add independent variable to model (length of interval).
            timer.makePrediciton(intervalInstances.numAttributes() - 1);
            timer.independantVariables.add(intervalInstances.numAttributes() - 1);

            //Build classifier with intervalInstances.
            if(classifier instanceof RandomTree){
                ((RandomTree)classifier).setKValue(intervalInstances.numAttributes() - 1);
            }
            baseClassifiers.add(AbstractClassifier.makeCopy(classifier));
            baseClassifiers.get(baseClassifiers.size()-1).buildClassifier(intervalInstances);

            //Add dependant variable to model (time taken).
            timer.dependantVariables.add(System.nanoTime() - timer.treeStartTime);

            //Serialise every 100 trees (if path has been set).
            if(treeCount % 100 == 0 && treeCount != 0 && serialisePath != null){
                saveToFile(seed);
            }
        }
        if (serialisePath != null) {
            saveToFile(seed);
        }

        if (timer.modelOutPath != null) {
            timer.saveModelToCSV(trainingData.relationName());
        }
        timer.forestTimeLimit += timer.forestElapsedTime;
        timer.forestElapsedTime = (System.nanoTime() - timer.forestStartTime);
        super.trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
        super.trainResults.setBuildTime(timer.forestElapsedTime);
        trainResults.setParas(getParameters());

    }

    private void findTrainAcc(Instances data){
        trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
        trainResults.setClassifierName(getClassifierName());
        trainResults.setDatasetName(data.relationName());
        trainResults.setFoldID(seed);
        int numTrees = 500;
        int bagProp = 100;
        int treeCount = 0;
        Classifier[] classifiers = new Classifier[numTrees];
        int[] timesInTest = new int[data.size()];
        double[][][] distributions = new double[numTrees][data.size()][(int)data.numClasses()];
        double[][] finalDistributions = new double[data.size()][(int)data.numClasses()];
        int[][] bags;
        ArrayList[] testIndexs = new ArrayList[numTrees];
        double[] bagAccuracies = new double[numTrees];

        this.setTrainTimeLimit(timer.forestTimeLimit, TimeUnit.NANOSECONDS);
        this.timer.forestTimeLimit = (long)((double)timer.forestTimeLimit * perForBag);

        bags = generateBags(numTrees, bagProp, data);


        for (; treeCount < numTrees && (System.nanoTime() - timer.forestStartTime) < (timer.forestTimeLimit - getTime()); treeCount++) {

            //Start tree timer.
            timer.treeStartTime = System.nanoTime();

            //Compute maximum interval length given time remaining.
            timer.buildModel();
            maxIntervalLength = (int)timer.getFeatureSpace((timer.forestTimeLimit) - (System.nanoTime() - (timer.forestStartTime - getTime())));

            Instances intervalInstances = produceIntervalInstancesUpdate(maxIntervalLength, data);

            intervalInstances = transformInstances(intervalInstances, transformType);

            //Add independent variable to model (length of interval).
            timer.makePrediciton(intervalInstances.numAttributes() - 1);
            timer.independantVariables.add(intervalInstances.numAttributes() - 1);

            Instances trainHeader = new Instances(intervalInstances, 0);
            Instances testHeader = new Instances(intervalInstances, 0);

            ArrayList<Integer> indexs = new ArrayList<>();
            for (int j = 0; j < bags[treeCount].length; j++) {
                if(bags[treeCount][j] == 0){
                    testHeader.add(intervalInstances.get(j));
                    timesInTest[j]++;
                    indexs.add(j);
                }
                for (int k = 0; k < bags[treeCount][j]; k++) {
                    trainHeader.add(intervalInstances.get(j));
                }
            }
            testIndexs[treeCount] = indexs;
            classifiers[treeCount] = new RandomTree();
            ((RandomTree)classifiers[treeCount]).setKValue(trainHeader.numAttributes() - 1);
            try {
                classifiers[treeCount].buildClassifier(trainHeader);
            } catch (Exception e) {
                e.printStackTrace();
            }
            for (int j = 0; j < testHeader.size(); j++) {
                try {
                    distributions[treeCount][indexs.get(j)] = classifiers[treeCount].distributionForInstance(testHeader.get(j));
                    if (classifiers[treeCount].classifyInstance(testHeader.get(j)) == testHeader.get(j).classValue()){
                        bagAccuracies[treeCount]++;
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
            bagAccuracies[treeCount] /= testHeader.size();
            trainHeader.clear();
            testHeader.clear();
            timer.dependantVariables.add(System.nanoTime() - timer.treeStartTime);
        }

        for (int i = 0; i < bags.length; i++) {
            for (int j = 0; j < bags[i].length; j++) {
                if(bags[i][j] == 0){
                    for (int k = 0; k < finalDistributions[j].length; k++) {
                        finalDistributions[j][k] += distributions[i][j][k];
                    }
                }
            }
        }

        for (int i = 0; i < finalDistributions.length; i++) {
            if (timesInTest[i] > 1){
                for (int j = 0; j < finalDistributions[i].length; j++) {
                    finalDistributions[i][j] /= timesInTest[i];
                }
            }
        }

        //Add to trainResults.
        double acc = 0.0;
        for (int i = 0; i < finalDistributions.length; i++) {
            double predClass = 0;
            double predProb = 0.0;
            for (int j = 0; j < finalDistributions[i].length; j++) {
                if (finalDistributions[i][j] > predProb){
                    predProb = finalDistributions[i][j];
                    predClass = j;
                }
            }
            trainResults.addPrediction(data.get(i).classValue(), finalDistributions[i], predClass, 0, "");
            if (predClass == data.get(i).classValue()){
                acc++;
            }
            trainResults.setAcc(acc / data.size());
        }
        this.timer.forestElapsedTime = System.nanoTime() - this.timer.forestStartTime;
    }

    private int[][] generateBags(int numBags, int bagProp, Instances data){
        int[][] bags = new int[numBags][data.size()];

        Random random = new Random(seed);
        for (int i = 0; i < numBags; i++) {
            for (int j = 0; j < data.size() * (bagProp/100.0); j++) {
                bags[i][random.nextInt(data.size())]++;
            }
        }
        return bags;
    }

    private Instances produceIntervalInstancesUpdate(int maxIntervalLength, Instances trainingData) {
        Instances intervalInstances;
        ArrayList<Attribute>attributes = new ArrayList<>();
        ArrayList<Integer> intervalAttIndexes = new ArrayList<>();

        startEndPoints.add(new int[2]);
        if(startEndPoints.size() == 1){
            startEndPoints.get(startEndPoints.size() - 1)[0] = 0;
            startEndPoints.get(startEndPoints.size() - 1)[1] = trainingData.numAttributes() - 1;
        }else{
            startEndPoints.get(startEndPoints.size() - 1)[0]=rand.nextInt((trainingData.numAttributes() - 1)- minIntervalLength);
            //This avoid calling nextInt(0)
            if(startEndPoints.get(startEndPoints.size() - 1)[0] == (trainingData.numAttributes() - 1) - 1 - minIntervalLength)
                startEndPoints.get(startEndPoints.size() - 1)[1] = trainingData.numAttributes() - 1 - 1;
            else{
                startEndPoints.get(startEndPoints.size() - 1)[1] = rand.nextInt((trainingData.numAttributes() - 1) - startEndPoints.get(startEndPoints.size() - 1)[0]);
                if(startEndPoints.get(startEndPoints.size() - 1)[1] < minIntervalLength)
                    startEndPoints.get(startEndPoints.size() - 1)[1] = minIntervalLength;
                startEndPoints.get(startEndPoints.size() - 1)[1] += startEndPoints.get(startEndPoints.size() - 1)[0];
            }
        }

        //int nearestPowerOfTwo = (int)FFT.MathsPower2.roundPow2((float) startEndPoints.get(startEndPoints.size() - 1)[1] - startEndPoints.get(startEndPoints.size() - 1)[0]);
        int nearestPowerOfTwo = startEndPoints.get(startEndPoints.size() - 1)[1] - startEndPoints.get(startEndPoints.size() - 1)[0];

        for (int i = 0; i < nearestPowerOfTwo; i ++) {
            //Attribute att = i + startEndPoints.get(startEndPoints.size() - 1)[0] < (startEndPoints.get(startEndPoints.size() - 1)[1] - startEndPoints.get(startEndPoints.size() - 1)[0]) ? trainingData.attribute(i + startEndPoints.get(startEndPoints.size() - 1)[0]) : new Attribute("att" + (i + 1 + startEndPoints.get(startEndPoints.size() - 1)[0]));
            Attribute att = i + startEndPoints.get(startEndPoints.size() - 1)[0] < trainingData.numAttributes() - 1 ? trainingData.attribute(i + startEndPoints.get(startEndPoints.size() - 1)[0]) : new Attribute("att" + (i + 1 + startEndPoints.get(startEndPoints.size() - 1)[0]));
            attributes.add(att);
        }

        attributes.add(trainingData.attribute(trainingData.numAttributes()-1));
        intervalInstances = new Instances(trainingData.relationName(), attributes, trainingData.size());
        double[] intervalInstanceValues = new double[nearestPowerOfTwo + 1];

        for (int i = 0; i < trainingData.size(); i++) {
            for (int j = 0; j < nearestPowerOfTwo; j++) {
                double value = j + startEndPoints.get(startEndPoints.size() - 1)[0] < trainingData.numAttributes() - 1 ? trainingData.get(i).value(j + startEndPoints.get(startEndPoints.size() - 1)[0]) : 0.0;
                //double value = j + startEndPoints.get(startEndPoints.size() - 1)[0] < (startEndPoints.get(startEndPoints.size() - 1)[1] - startEndPoints.get(startEndPoints.size() - 1)[0]) ? trainingData.get(i).value(j + startEndPoints.get(startEndPoints.size() - 1)[0]) : 0.0;
                intervalInstanceValues[j] = value;
            }

            DenseInstance intervalInstance = new DenseInstance(intervalInstanceValues.length);
            intervalInstance.replaceMissingValues(intervalInstanceValues);
            intervalInstance.setValue(intervalInstanceValues.length-1, trainingData.get(i).classValue());
            intervalInstances.add(intervalInstance);
        }

        intervalInstances.setClassIndex(intervalInstances.numAttributes() - 1);

        return intervalInstances;
    }

    /**
     * Classify one instance from test set.
     * @param instance the instance to be classified
     * @return double representing predicted class of test instance.
     * @throws Exception
     */
    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double[]distribution = distributionForInstance(instance);

        int maxVote=0;
        for(int i = 1; i < distribution.length; i++)
            if(distribution[i] > distribution[maxVote])
                maxVote = i;
        return maxVote;
    }

    /**
     * Distribution or probabilities over classes for one test instance.
     * @param testInstance
     * @return double array of size numClasses containing probabilities of test instance belonging to each class.
     * @throws Exception
     */
    @Override
    public double[] distributionForInstance(Instance testInstance) throws Exception {
        double[]distribution = new double[testInstance.numClasses()];
        //ArrayList<Attribute> attributes;

        //For every base classifier.
        for (int i = 0; i < baseClassifiers.size(); i++) {
            //Transform instance into interval instance.
            /*attributes = new ArrayList<>();
            for (int j = 0; j < intervalsAttIndexes.get(i).size(); j++) {
                attributes.add(testInstance.attribute(intervalsAttIndexes.get(i).get(j)));
            }
            attributes.add(testInstance.attribute(testInstance.numAttributes() - 1));
            Instances instances = new Instances("relationName", attributes, 1);
            Instance intervalInstance = produceIntervalInstance(testInstance, i);
            instances.add(intervalInstance);
            instances.setClassIndex(intervalInstance.numAttributes() - 1);*/


            Instance intervalInstance = null;
            //Transform interval instance into PS, ACF, ACF_PS or ACF_PS_AR
            if (transformType != null) {
                try{
                    intervalInstance = transformInstances(produceIntervalInstanceUpdate(testInstance, i), transformType).firstInstance();
                }catch(Exception e){
                    intervalInstance = transformInstances(produceIntervalInstanceUpdate(testInstance, i), transformType).firstInstance();
                }
            }

           /* double[] temp = null;
            try{
                temp = baseClassifiers.get(i).distributionForInstance(intervalInstance);
            }catch(Exception e){
                temp = baseClassifiers.get(i).distributionForInstance(intervalInstance);
            }*/
            /*for (int j = 0; j < testInstance.numClasses(); j++) {
                distribution[j] += temp[j];
            }*/
            distribution[(int)baseClassifiers.get(i).classifyInstance((intervalInstance))]++;
        }
        for (int j = 0; j < testInstance.numClasses(); j++) {
            distribution[j] /= baseClassifiers.size();
        }
        return distribution;
    }

    /**
     * Returns default capabilities of the classifier. These are that the
     * data must be numeric, with no missing and a nominal class
     * @return the capabilities of this classifier
     **/
    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();
        // attributes must be numeric
        // Here add in relational when ready
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        // instances
        result.setMinimumNumberInstances(1);
        return result;


    }

    /**
     * Method returning all classifier parameters as a string.
     * for EnhancedAbstractClassifier
     * @return
     */
    @Override
    public String getParameters() {

        String result = "Total Time Taken," + timer.forestElapsedTime
                + ", Contract Length (ns), " + timer.forestTimeLimit
                + ", Percentage contract for OOB, " + perForBag
                + ", NumAtts," + data.numAttributes()
                + ", MaxNumTrees," + numClassifiers
                + ", NumTrees," + treeCount
                + ", MinIntervalLength," + minIntervalLength
                + ", Filters, " + this.transformType.toString()
                + ", Final Coefficients (time = a * x^2 + b * x + c)"
                + ", a, " + timer.a
                + ", b, " + timer.b
                + ", c, " + timer.c;
        return result;
    }

    /**
     * for interface TrainTimeEstimate
     * @param amount: time in nanoseconds
     */
    @Override
    public void setTrainTimeLimit(long amount) {
        timer.setTimeLimit(amount);
    }

    /**
     * for interface TechnicalInformationHandler
     * @return info on paper
     */
    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation    result;
        result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
        result.setValue(TechnicalInformation.Field.AUTHOR, "US");
        result.setValue(TechnicalInformation.Field.YEAR, "2019?");
        result.setValue(TechnicalInformation.Field.TITLE, "title");
        result.setValue(TechnicalInformation.Field.JOURNAL, "LNCS");
        result.setValue(TechnicalInformation.Field.VOLUME, "XX");
        result.setValue(TechnicalInformation.Field.PAGES, "XXXXX");
        return result;
    }

    /**
     * for interface Tuneable
     * @return
     */
    @Override
    public ParameterSpace getDefaultParameterSearchSpace(){
        ParameterSpace ps=new ParameterSpace();
        String[] numTrees={"100","200","300","400","500","600","700","800","900","1000"};
        ps.addParameter("-T", numTrees);
//Add others here
        return ps;
    }
    /**
     * Parses a given list of options to set the parameters of the classifier.
     * We use this for the tuning mechanism, setting parameters through setOptions
     <!-- options-start -->
     * Valid options are: <p/>
     * <pre> -T
     * Number of trees.</pre>
     *
     * <pre> -I
     * Number of intervals to fit.</pre>
     *
     <!-- options-end -->
     *
     * @param options the list of options as an array of strings
     * @throws Exception if an option is not supported
     */
    @Override
    public void setOptions(String[] options) throws Exception{
        String numTreesString=Utils.getOption('T', options);
        if (numTreesString.length() != 0)
            numClassifiers = Integer.parseInt(numTreesString);
/**
 * INSERT OTHER OPTIONS HERE
 */

    }


    /**
     * Private inner class containing all logic pertaining to timing.
     * CRISE is contracted via updating a linear regression model (y = a * x^2 + b * x + c) in which the dependant
 variable (y) is time taken and the independent variable (x) is interval length.
 The equation is then reordered to solve for positive x, providing the upper bound on the interval space.
 Dividing this by minNumtrees - treeCount gives the maximum space such that in the worse case the contract is met.
     */
    private class Timer implements Serializable{

        protected long forestTimeLimit = Long.MAX_VALUE;
        protected long forestStartTime = 0;
        protected long treeStartTime = 0;
        protected long forestElapsedTime = 0;

        protected ArrayList<Integer> independantVariables = null;
        protected ArrayList<Long> dependantVariables = null;
        protected ArrayList<Double> predictions = null;
        private ArrayList<Double> aValues = null;
        private ArrayList<Double> bValues = null;
        private ArrayList<Double> cValues = null;

        protected double a = 0.0;
        protected double b = 0.0;
        protected double c = 0.0;

        protected String modelOutPath = null;

        /**
         * Called in CRISE.initialise in order to reset timer.
         */
        protected void reset(){
            independantVariables = new ArrayList<>();
            dependantVariables = new ArrayList<>();
            predictions = new ArrayList<>();
            aValues = new ArrayList<>();
            bValues = new ArrayList<>();
            cValues = new ArrayList<>();
        }

        /**
         * computes coefficients (a, b, c).
         */
        protected void buildModel(){

            a = 0.0;
            b = 0.0;
            c = 0.0;
            double numberOfVals = (double) independantVariables.size();
            double smFrstScrs = 0.0;
            double smScndScrs = 0.0;
            double smSqrFrstScrs = 0.0;
            double smCbFrstScrs = 0.0;
            double smPwrFrFrstScrs = 0.0;
            double smPrdtFrstScndScrs = 0.0;
            double smSqrFrstScrsScndScrs = 0.0;

            for (int i = 0; i < independantVariables.size(); i++) {
                smFrstScrs += independantVariables.get(i);
                smScndScrs += dependantVariables.get(i);
                smSqrFrstScrs += Math.pow(independantVariables.get(i), 2);
                smCbFrstScrs += Math.pow(independantVariables.get(i), 3);
                smPwrFrFrstScrs += Math.pow(independantVariables.get(i), 4);
                smPrdtFrstScndScrs += independantVariables.get(i) * dependantVariables.get(i);
                smSqrFrstScrsScndScrs += Math.pow(independantVariables.get(i), 2) * dependantVariables.get(i);
            }

            double valOne = smSqrFrstScrs - (Math.pow(smFrstScrs, 2) / numberOfVals);
            double valTwo = smPrdtFrstScndScrs - ((smFrstScrs * smScndScrs) / numberOfVals);
            double valThree = smCbFrstScrs - ((smSqrFrstScrs * smFrstScrs) / numberOfVals);
            double valFour = smSqrFrstScrsScndScrs - ((smSqrFrstScrs * smScndScrs) / numberOfVals);
            double valFive = smPwrFrFrstScrs - (Math.pow(smSqrFrstScrs, 2) / numberOfVals);

            a = ((valFour * valOne) - (valTwo * valThree)) / ((valOne * valFive) - Math.pow(valThree, 2));
            b = ((valTwo * valFive) - (valFour * valThree)) / ((valOne * valFive) - Math.pow(valThree, 2));
            c = (smScndScrs / numberOfVals) - (b * (smFrstScrs / numberOfVals)) - (a * (smSqrFrstScrs / numberOfVals));

            aValues.add(a);
            bValues.add(b);
            cValues.add(c);
        }

        /**
         * Adds x(y') to predictions arrayList for model output.
         * @param x interval size.
         */
        protected void makePrediciton(int x){
            predictions.add(a * Math.pow(x, 2) + b * x + c);
        }


        /**
         * Given time remaining returns largest interval space possible.
         * Takes into account whether minNumTrees is satisfied.
         * ensures minIntervalLength < x < maxIntervalLength.
         * @param timeRemaining
         * @return interval length
         */
        protected double getFeatureSpace(long timeRemaining){
            double y = timeRemaining;
            double x = ((-b) + (Math.sqrt((b * b) - (4 * a * (c - y))))) / (2 * a);

            if (treeCount < minNumTrees) {
                x = x / (minNumTrees - treeCount);
            }
            if(treeCount == minNumTrees){
                maxIntervalLength = data.numAttributes()-1;
            }

            if (x > maxIntervalLength || Double.isNaN(x)) {
                x = maxIntervalLength;
            }
            if(x < minIntervalLength){
                x = minIntervalLength;
            }

            return x;
        }

        /**
         *
         * @param timeLimit in nano seconds
         */
        protected void setTimeLimit(long timeLimit){
            this.forestTimeLimit = timeLimit;
        }

        protected void printModel(){

            for (int i = 0; i < independantVariables.size(); i++) {
                System.out.println(Double.toString(independantVariables.get(i)) + "," + Double.toString(dependantVariables.get(i)) + "," + Double.toString(predictions.get(i)));
            }
        }

        protected void saveModelToCSV(String problemName){
            try{
                FullAccessOutFile outFile = new FullAccessOutFile((modelOutPath.isEmpty() ? "timingModel" + (int) seed + ".csv" : modelOutPath + "/" + problemName + "/" + "/timingModel" + (int) seed + ".csv"));
                for (int i = 0; i < independantVariables.size(); i++) {
                    outFile.writeLine(Double.toString(independantVariables.get(i)) + ","
                            + Double.toString(dependantVariables.get(i)) + ","
                            + Double.toString(predictions.get(i)) + ","
                            + Double.toString(timer.aValues.get(i)) + ","
                            + Double.toString(timer.bValues.get(i)) + ","
                            + Double.toString(timer.cValues.get(i)));
                }
                outFile.closeFile();
            }catch(Exception e){
                System.out.println("Mismatch between relation name and name of results folder: " + e);
            }

        }
    }

    public static void main(String[] args){

        Instances dataTrain = loadDataNullable("Z:/ArchiveData/Univariate_arff" + "/" + DatasetLists.newProblems27[2] + "/" + DatasetLists.newProblems27[2] + "_TRAIN");
        Instances dataTest = loadDataNullable("Z:/ArchiveData/Univariate_arff" + "/" + DatasetLists.newProblems27[2] + "/" + DatasetLists.newProblems27[2] + "_TEST");
        Instances data = dataTrain;
        data.addAll(dataTest);

        ClassifierResults cr = null;
        SingleSampleEvaluator sse = new SingleSampleEvaluator();
        sse.setPropInstancesInTrain(0.5);
        sse.setSeed(1);

        RISE RISE = null;
        System.out.println("Dataset name: " + data.relationName());
        System.out.println("Numer of cases: " + data.size());
        System.out.println("Number of attributes: " + (data.numAttributes() - 1));
        System.out.println("Number of classes: " + data.classAttribute().numValues());
        System.out.println("\n");
        try {
            RISE = new RISE();
            RISE.setSavePath("D:/Test/Testing/Serialising/");
            //cRISE.setTrainTimeLimit(TimeUnit.MINUTES, 5);
            RISE.setTransformType(TransformType.ACF_FFT);
            cr = sse.evaluate(RISE, data);
            System.out.println("FFT");
            System.out.println("Accuracy: " + cr.getAcc());
            System.out.println("Build time (ns): " + cr.getBuildTimeInNanos());
        } catch (Exception e) {
            e.printStackTrace();
        }

        try {
            ClassifierResults temp = ClassifierTools.testUtils_evalOnIPD(RISE);
            temp.writeFullResultsToFile("D:\\Test\\Testing\\TestyStuff\\cRISE.csv");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

/*
Dataset = ADIAC
With reload (@ 200 trees)
    Accuracy: 0.7868020304568528
    Build time (ns): 60958242098

With reload (@ 500 trees (Completed build))
    Accuracy: 0.7868020304568528
    Build time (ns): 8844999832

With no reload but serialising at 100 intervals.
    Accuracy: 0.7868020304568528
    Build time (ns): 96078716938

No serialising
    Accuracy: 0.7868020304568528
    Build time (ns): 88964973765
*/

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
package machine_learning.classifiers.ensembles;

import experiments.CollateResults;
import experiments.data.DatasetLists;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import machine_learning.classifiers.ensembles.voting.MajorityVote;
import machine_learning.classifiers.ensembles.weightings.EqualWeighting;
import evaluation.storage.ClassifierResults;
import experiments.Experiments;
import experiments.data.DatasetLoading;
import tsml.classifiers.EnhancedAbstractClassifier;
import static utilities.GenericTools.indexOfMax;
import utilities.InstanceTools;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.filters.Filter;

/**
 * Implementation of ensemble selection
 * 
 *  @inproceedings{caruana2004ensemble,
 *    title={Ensemble selection from libraries of models},
 *    author={Caruana, Rich and Niculescu-Mizil, Alexandru and Crew, Geoff and Ksikes, Alex},
 *    booktitle={Proceedings of the twenty-first international conference on Machine learning},
 *    pages={18},
 *    year={2004},
 *    organization={ACM}
 *  }
 * 
 * 
 * Built on top of hesca for it's classifierresults file building/handling capabilities.
 * In this relatively naive implementation, the ensemble after build classifier still actually has the entire library in it,
 * however one or more of the models may have a (PRIOR) weighting of 0
 * For the purposes we will be using this for (with something on the order of a couple dozen classifiers at most) this will work fine 
 * in terms of runtime etc. 
 * 
 * However in the future refactors for optimisation purposes may occur if e.g we intend to handle much larger libraries (e.g, ensembling 
 * over large para-space searched results)
 * 
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class EnsembleSelection extends CAWPE {

    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation 	result;
        result = new TechnicalInformation(TechnicalInformation.Type.INPROCEEDINGS);
        result.setValue(TechnicalInformation.Field.AUTHOR, "R. Caruana, A. Niculescu-Mizil, G. Crew and A. Ksikes");
        result.setValue(TechnicalInformation.Field.YEAR, "2004");
        result.setValue(TechnicalInformation.Field.TITLE, "Ensemble selection from libraries of models");
        result.setValue(TechnicalInformation.Field.BOOKTITLE, "Proceedings of the twenty-first international conference on Machine learning");
        result.setValue(TechnicalInformation.Field.PAGES, "18");
        result.setValue(TechnicalInformation.Field.ORGANIZATION, "ACM");

        return result;
    }
    
    
//    Integer numBags = null; //default 2 * floor(log(sizeOfLibrary)), i.e 22 classifiers gives 8 bags. Paper says 20 bags from 2000 models, so definitely seems fair
    Integer numBags = null; //default 10. Paper says 20 bags from 2000 models, so definitely seems fair
    Double propOfModelsInEachBag = null; //aka p, default 0.5. value used through exps in paper, though some suggestion that p around 0.1 to 0.3 may be better, future work
    Integer numOfTopModelsToInitialiseBagWith = null; //aka N, default value set to 2 for now, since only 22 classifiers being used atm. paper suggested around 5-25 for library of 2000 models
    
    //we currently intend to use only a library of 22 classifiers, from which we'll sample 11 (p=0.5). 100 models is more than any sampling with replacement 
    //run should take, but jsut as a safeguard against minutely incrementing accuracy because of double precision shenanigans, have this in as a second stopping condition
    final int MAX_SUBENSEMBLE_SIZE = 100; 
    
    Random rng;

    public EnsembleSelection() {
        super(); //sets default classifiers etc 
        
        //overwriting relevant parts 
        ensembleName = "EnsembleSelection"; 
//        votingScheme = new MajorityConfidence();
        votingScheme = new MajorityVote();
        weightingScheme = new EqualWeighting();
        
        rng = new Random(0);
    }
    
    public Integer getNumBags() {
        return numBags;
    }

    public void setNumBags(Integer numBags) {
        this.numBags = numBags;
    }

    public Double getPropOfModelsInEachBag() {
        return propOfModelsInEachBag;
    }

    public void setPropOfModelsInEachBag(Double propOfModelsInEachBag) {
        this.propOfModelsInEachBag = propOfModelsInEachBag;
    }

    public Integer getNumOfTopModelsToInitialiseBagWith() {
        return numOfTopModelsToInitialiseBagWith;
    }

    public void setNumOfTopModelsToInitialiseBagWith(Integer numOfTopModelsToInitialiseBagWith) {
        this.numOfTopModelsToInitialiseBagWith = numOfTopModelsToInitialiseBagWith;
    }
    
    @Override
    public void setSeed(int seed) {
        super.setSeed(seed);
        rng = new Random(seed);
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        printlnDebug("**EnsembleSelection TRAIN**");
        
        
        //housekeeping
        if (resultsFilesParametersInitialised) {
            if (readResultsFilesDirectories.length > 1)
                if (readResultsFilesDirectories.length != modules.length)
                    throw new Exception("EnsembleSelection.buildClassifier: more than one results path given, but number given does not align with the number of classifiers/modules.");

            if (writeResultsFilesDirectory == null)
                writeResultsFilesDirectory = readResultsFilesDirectories[0];
        }
        
        long startTime = System.nanoTime();
        
        //transform data if specified
        if(this.transform==null){
            this.trainInsts = new Instances(data);
        }else{
            this.trainInsts = transform.transform(data); //TODO: this could call fit?
        }
        
        //init
        this.numTrainInsts = trainInsts.numInstances();
        this.numClasses = trainInsts.numClasses();
        this.numAttributes = trainInsts.numAttributes();
        
        //set up modules
        initialiseModules();
        
        //these won't actually do anything at this stage, except some basic initialisation
        //mostly still calling these as a relic from hesca for potential future proofing
        weightingScheme.defineWeightings(modules, numClasses);
        votingScheme.trainVotingScheme(modules, numClasses);
        
//NOW THE ACTUAL SELECTION STUFF
        //have 'library' of models size L
        //init: list of sub ensembles
        //for each 'bag' in the bagging of ensemble stage (b in B ?)
            //init model set with p*L random models, is fraction of models in each bag
            //initialise this sub-ensemble with top N classifiers in this bag (accuracy only for now)
            //calc initial ensemble accuracy (even weighting over models)
            //while acc is improving
                //test the addition of *each* model to the ensemble-so-far
                //if an increase in ensemble performance can be acheived, add the model that gives the biggest to the ensemble
                //can play with the prior weights to simulate additional models of the same kind being added? (selection with replacement)
            //add finalised sub-ensemble to the bag of ensembles
        //now have a set of ensembles, which are effectively each a weighted average of the base classifiers (weighted by the number of tiems there were included)
        //so now jsut average over these ensembles again to get the final weighted ensemble
        
        //a lot of this code could be easily refactored for efficiency, however will leave it as is (a pretty naive implementation)
        //for ease of understanding/maintenance

        //init the params if not already set by user
        if (numBags == null)
//            numBags = (int) (Math.log(modules.length) / Math.log(2)) * 2;
            numBags = 10;
        if (propOfModelsInEachBag == null)
            propOfModelsInEachBag = .5;
        if (numOfTopModelsToInitialiseBagWith == null)
            numOfTopModelsToInitialiseBagWith = 1; // log(sizeOfBag) ? 
        
        int numModelsInEachBag = Math.max(1, (int)Math.round(propOfModelsInEachBag * modules.length));
        
        //will hold the actual ensembles as they go along
        List<List<EnsembleModule>> subensembles = new ArrayList<>(numBags);
        ClassifierResults globalEnsembleResults = null;
        
        for (int bagID = 0; bagID < numBags; bagID++) {
            List<EnsembleModule> bagOfModels = sample(modules, numModelsInEachBag);
            //todo check this, treeset should do sorting for us, however unsure if ordering is maintined durign toarray()
            
            List<EnsembleModule> subensemble = new ArrayList<>();
            
            ClassifierResults subEnsembleResults = null;
            
            if (numOfTopModelsToInitialiseBagWith!=null && numOfTopModelsToInitialiseBagWith > 0) {
                int lastInd = bagOfModels.size()-1;
                EnsembleModule model = bagOfModels.get(lastInd); //best in trainEstimator
                subensemble.add(model);
                subEnsembleResults = model.trainResults;
                
                for (int i = 1; i < numOfTopModelsToInitialiseBagWith; i++) {
                    model = bagOfModels.get(lastInd - i); //next highest trainEstimator score
                    subensemble.add(model);
                    
                    subEnsembleResults = combinePredictions(subEnsembleResults, i, model.trainResults);
                }
            }
                    
            //initialisation of subensemble done, start the forward selection
            double accSoFar;
            double newAcc = subEnsembleResults == null ? .0 : subEnsembleResults.getAcc();
            boolean finished;
            do {
                finished = true;
                accSoFar = newAcc;
                
                ClassifierResults[] candidateResults = new ClassifierResults[bagOfModels.size()];
                double[] accs = new double[bagOfModels.size()];
                for (int modelID = 0; modelID < bagOfModels.size(); modelID++) {
                    candidateResults[modelID] = combinePredictions(subEnsembleResults, subensemble.size(), bagOfModels.get(modelID).trainResults);
                    accs[modelID] = candidateResults[modelID].getAcc();
                }
                
                int maxAccInd = (int)utilities.GenericTools.indexOfMax(accs);
                newAcc = accs[maxAccInd];
                
                if (newAcc > accSoFar) {
                    finished = false;
                    subEnsembleResults = candidateResults[maxAccInd];
                    subensemble.add(bagOfModels.get(maxAccInd));
                    
                    if (subensemble.size() >= MAX_SUBENSEMBLE_SIZE)
                        finished = true;
                } 
                
            } while (!finished);
                
            subensembles.add(subensemble);
            if (globalEnsembleResults == null)
                globalEnsembleResults = subEnsembleResults;
            else
                globalEnsembleResults = combinePredictions(globalEnsembleResults, bagID, subEnsembleResults);
        }
        
        //have sub ensembles, now to produce the final weighted ensemble
        
        //i think easiest way is to just continue using the hesca architecture via 
        //'equalweighting' and majorityconfidence, but fix the PRIOR weights to the abundance of the classifiers here
        
        //init all modules to have prior weight of 0.0
        for (EnsembleModule module : modules)
            module.priorWeight = 0.0;
        
        //for all models in all subsembles, increment that model's prior weight in the final ensemble essentially
        for (List<EnsembleModule> subensemble : subensembles) {
            for (EnsembleModule model : subensemble) {
                int ind = 0;
                for ( ; ind < modules.length; ind++)
                    if (model == modules[ind]) //by reference should work
                        break;
                assert(ind != modules.length);
                
                modules[ind].priorWeight++;
            }
        }
//END OF THE ACTUAL SELECTION STUFF   

        trainResults = globalEnsembleResults;
        trainResults.setClassifierName("EnsembleSelection");
        trainResults.setDatasetName(datasetName);
        trainResults.setFoldID(seed);
        trainResults.setSplit("train");
        
        long buildTime = System.nanoTime() - startTime;
        if (readIndividualsResults) {
            //we need to sum the modules' reported build time as well as the weight
            //and voting definition time
            for (EnsembleModule module : modules) {
                buildTime += module.trainResults.getBuildTimeInNanos();
                
                //TODO see other todo in trainModules also. Currently working under 
                //assumption that the estimate time is already accounted for in the build
                //time of TrainAccuracyEstimators, i.e. those classifiers that will 
                //estimate their own accuracy during the normal course of training
                if (!EnhancedAbstractClassifier.classifierIsEstimatingOwnPerformance(module.getClassifier()))
                    buildTime += module.trainResults.getErrorEstimateTime();
            }
        }
        
        trainResults.setBuildTime(buildTime); //store the buildtime to be saved
        
        this.testInstCounter = 0; //prep for start of testing
    }
    
    public List<EnsembleModule> sample(final EnsembleModule[] pool, int numToPick) {
        //todo refactor...
        LinkedList<EnsembleModule> pooll = new LinkedList<>();
        for (EnsembleModule module : pool)
            pooll.add(module);
        
        List<EnsembleModule> res = new ArrayList<>(numToPick);
        
        for (int i = 0; i < numToPick; i++) {
            int toRemove = rng.nextInt(pooll.size());
            res.add(pooll.remove(toRemove));
        }
        return res;
    }
    
    public static class SortByTrainAcc implements Comparator<EnsembleModule> {
        @Override
        public int compare(EnsembleModule o1, EnsembleModule o2) {
            return Double.compare(o1.trainResults.getAcc(), o2.trainResults.getAcc());
        }
    }
    
    public ClassifierResults combinePredictions(final ClassifierResults ensembleSoFarResults, int ensembleSizeSoFar, final ClassifierResults newModelResults) throws Exception {
        ClassifierResults newResults = new ClassifierResults(numClasses);
        assert(ensembleSoFarResults.getTimeUnit().equals(newModelResults.getTimeUnit()));
        newResults.setTimeUnit(ensembleSoFarResults.getTimeUnit());
        
        for (int inst = 0; inst < ensembleSoFarResults.getProbabilityDistributions().size(); inst++) {
            double[] ensDist = ensembleSoFarResults.getProbabilityDistribution(inst);
            double[] indDist = newModelResults.getProbabilityDistribution(inst);
            
            assert(ensDist.length == numClasses);
            assert(indDist.length == numClasses);
            
            double[] newDist = new double[numClasses];
            for (int c = 0; c < numClasses; c++)
                newDist[c] = ((ensDist[c] * ensembleSizeSoFar) + indDist[c]) / (ensembleSizeSoFar+1); //expand existing average, add in new model, and divide again
            
            //todo: exactly how to time train-instance predictions for this classifier is very debatable. going with this for now
            long predTime = ensembleSoFarResults.getPredictionTime(inst) + newModelResults.getPredictionTime(inst);
            newResults.addPrediction(newDist, indexOfMax(newDist), predTime, "");
        }
        
        newResults.finaliseResults(ensembleSoFarResults.getTrueClassValsAsArray());
        return newResults;
    }
    
    public static void main(String[] args) throws Exception {
 //       tests();
//        ana();
    }
   
    public static void tests() { 
        String resPath = "C:/JamesLPHD/HESCA/UCI/UCIResults/";
        int numfolds = 30;
        
        String[] dsets = DatasetLists.UCIContinuousFileNames;
//        String[] skipDsets = new String[] { "adult", "chess-krvk", "chess-krvkp", "connect-4", "miniboone", };
                
//        String[] dsets = new String[] { "hayes-roth" };
        String[] skipDsets = new String[] { };
        
        String classifier = "EnsembleSelectionAll22Classifiers_Preds";
        
        for (String dset : dsets) {          
            if (Arrays.asList(skipDsets).contains(dset))
                continue;
            
            System.out.println(dset);
            
            Instances all = DatasetLoading.loadDataNullable("C:/UCI Problems/" + dset + "/" + dset + ".arff");
            
            for (int fold = 0; fold < numfolds; fold++) {
                String predictions = resPath+classifier+"/Predictions/"+dset;
                File f=new File(predictions);
                if(!f.exists())
                    f.mkdirs();
        
                //Check whether fold already exists, if so, dont do it, just quit
                if(!CollateResults.validateSingleFoldFile(predictions+"/testFold"+fold+".csv")){
                    Instances[] data = InstanceTools.resampleInstances(all, fold, .5);
                    
                    EnsembleSelection c = new EnsembleSelection();
                    
                    //for full kitchen sink classifier list, set the init # models to 2 from each bag, there's only 22 total, so 11 in each bag
                    c.setClassifiers(null, CAWPE_bigClassifierList, null);
                    c.setNumOfTopModelsToInitialiseBagWith(2);
                    
                    //for just the hesca models, use default of 1
//                    c.setClassifiers(null, PAPER_HESCA, null);
//                    c.setClassifiers(null, CAWPE_MajorityVote.HESCAplus_V4_Classifiers, null);
                    
                    c.setBuildIndividualsFromResultsFiles(true);
                    c.setResultsFileLocationParameters(resPath, dset, fold);
                    c.setSeed(fold);
                    c.setEstimateOwnPerformance(true);
                    c.setResultsFileWritingLocation(resPath);
                                        
                    Experiments.ExperimentalArguments exp = new Experiments.ExperimentalArguments();
                    exp.classifierName = classifier;
                    exp.datasetName = dset;
                    exp.foldId = fold;
                    exp.generateErrorEstimateOnTrainSet = true;
                    exp.testFoldFileName = predictions+"/testFold"+fold+".csv";
                    exp.trainFoldFileName = predictions+"/trainFold"+fold+".csv";
//                        exp.performTimingBenchmark = true;
                    Experiments.runExperiment(exp,data[0],data[1],c);
                }
            }
        }
    }
   

    public static String[] CAWPE_basic = new String[] { 
        "NN",
        "SVML",
        "C4.5",
        "Logistic",
        "MLP"
    };
    
    public static String[] CAWPE_bigClassifierList= new String[] { 
        //original 
        "RotFDefault", 
        "RandF",
        "SVMQ",
        "NN",
        "SVML",
        "C4.5",
        "NB",
        "bayesNet",
        
        //homoensembles
        "DaggingDefault",
        "MultiBoostABDefault",
        "AdaBoostM1Default",
        "BaggingDefault",
        "LogitBoostDefault",
        "DecorateDefault",
        "ENDDefault",
        "RandomCommitteeDefault",
        
        //extra
        "Logistic",
        "MLP",   
        "DNN",
        "1NN",
        "DecisionTable",
        "REPTree",        
    };
}

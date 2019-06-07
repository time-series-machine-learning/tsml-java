package classifiers.distance_based.elastic_ensemble;

import classifiers.distance_based.elastic_ensemble.selection.BestPerTypeSelector;
import classifiers.distance_based.elastic_ensemble.selection.Selector;
import classifiers.distance_based.knn.Knn;
import classifiers.template_classifier.TemplateClassifier;
import distances.derivative_time_domain.ddtw.CachedDdtw;
import distances.time_domain.dtw.Dtw;
import distances.derivative_time_domain.wddtw.CachedWddtw;
import distances.time_domain.erp.Erp;
import distances.time_domain.lcss.Lcss;
import distances.time_domain.msm.Msm;
import distances.time_domain.twe.Twe;
import distances.time_domain.wdtw.Wdtw;
import evaluation.storage.ClassifierResults;
import evaluation.tuning.ParameterSpace;
import timeseriesweka.classifiers.ensembles.EnsembleModule;
import timeseriesweka.classifiers.ensembles.voting.MajorityVoteByConfidence;
import timeseriesweka.classifiers.ensembles.voting.ModuleVotingScheme;
import timeseriesweka.classifiers.ensembles.weightings.ModuleWeightingScheme;
import timeseriesweka.classifiers.ensembles.weightings.TrainAcc;
import utilities.Utilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;
import java.util.function.Function;

public class ElasticEnsemble extends TemplateClassifier {

    public static List<Function<Instances, ParameterSpace>> getClassicParameterSpaceGetters() {
        return new ArrayList<>(Arrays.asList(
            instances -> Dtw.euclideanParameterSpace(),
            instances -> Dtw.fullWindowParameterSpace(),
            Dtw::discreteParameterSpace,
            instances -> CachedDdtw.fullWindowParameterSpace(),
            CachedDdtw::discreteParameterSpace,
            instances -> Wdtw.discreteParameterSpace(),
            instances -> CachedWddtw.discreteParameterSpace(),
            Lcss::discreteParameterSpace,
            Erp::discreteParameterSpace,
            instances -> Msm.discreteParameterSpace(),
            instances -> Twe.discreteParameterSpace()
            ));
    }

    public static List<Function<Instances, ParameterSpace>> getDefaultParameterSpaceGetters() {
        return new ArrayList<>(Arrays.asList(
            Dtw::allDiscreteParameterSpace,
            CachedDdtw::allDiscreteParameterSpace,
            instances -> Wdtw.discreteParameterSpace(),
            instances -> CachedWddtw.discreteParameterSpace(),
            Lcss::discreteParameterSpace,
            Erp::discreteParameterSpace,
            instances -> Msm.discreteParameterSpace(),
            instances -> Twe.discreteParameterSpace()
                                            ));
    }

    public ElasticEnsemble() {
        this(getDefaultParameterSpaceGetters());
    }

    public ElasticEnsemble(Function<Instances, ParameterSpace>... parameterSpaceGetters) {
        this(Arrays.asList(parameterSpaceGetters));
    }

    private final List<Function<Instances, ParameterSpace>> parameterSpaceGetters = new ArrayList<>();
    private final List<ParameterSpace> parameterSpaces = new ArrayList<>();

    public ElasticEnsemble(List<Function<Instances, ParameterSpace>> parameterSpaceGetters) {
        setParameterSpaceGetters(parameterSpaceGetters);
    }

    public void setParameterSpaceGetters(List<Function<Instances, ParameterSpace>> parameterSpaceGetters) {
        this.parameterSpaceGetters.clear();
        this.parameterSpaceGetters.addAll(parameterSpaceGetters);
    }

    public static List<ParameterSpace> getParameterSpaces(Instances instances, List<Function<Instances, ParameterSpace>> parameterSpaceGetters) {
        List<ParameterSpace> parameterSpaces = new ArrayList<>();
        for(Function<Instances, ParameterSpace> getter : parameterSpaceGetters) {
            ParameterSpace parameterSpace = getter.apply(instances);
            parameterSpaces.add(parameterSpace);
        }
        return parameterSpaces;
    }

    private boolean removeDuplicateParameterValues = true;
    private EnsembleModule[] modules = null;
    private ModuleWeightingScheme weightingScheme = new TrainAcc();
    private ModuleVotingScheme votingScheme = new MajorityVoteByConfidence();
    private long phaseTime = 0;
    private final List<Knn> knns = new ArrayList<>();
    private final List<Iterator<String[]>> parameterSetIterators = new ArrayList<>();
    private Selector<Candidate> selector = new BestPerTypeSelector<>(candidate -> candidate.getKnn()
                                                                                           .getDistanceMeasure()
                                                                                           .toString(), (candidate, t1) -> Comparator.comparingDouble(ClassifierResults::getAcc).compare(candidate.getTrainResults(), t1.getTrainResults()));

    @Override
    public void buildClassifier(final Instances trainInstances) throws
                                                      Exception {
        long startTime = System.nanoTime();
        Random random = getTrainRandom();
        knns.clear();
        selector.setRandom(random);
        parameterSetIterators.clear();
        parameterSpaces.clear();
        parameterSpaces.addAll(getParameterSpaces(trainInstances, parameterSpaceGetters));
        if(removeDuplicateParameterValues) {
            for(ParameterSpace parameterSpace : parameterSpaces) {
                parameterSpace.removeDuplicateValues();
            }
        }
        for(ParameterSpace parameterSpace : parameterSpaces) {
            Iterator<String[]> iterator = new ParameterSetIterator(parameterSpace, new RandomIndexIterator(random, parameterSpace.size()));
            if(iterator.hasNext()) parameterSetIterators.add(iterator);
        }
        incrementTrainTimeNanos(System.nanoTime() - startTime);
        boolean remainingParameters = !parameterSetIterators.isEmpty();
        boolean remainingKnns = !knns.isEmpty();
        while((remainingParameters || remainingKnns) && remainingTrainContract() > phaseTime) {
            long startPhaseTime = System.nanoTime();
            Knn knn;
            boolean choice = true;
            if(remainingParameters && remainingKnns) {
                choice = random.nextBoolean();
            }
            if(choice) {
                int index = random.nextInt(parameterSetIterators.size());
                Iterator<String[]> iterator = parameterSetIterators.get(index);
                String[] parameters = iterator.next();
                if(!iterator.hasNext()) {
                    parameterSetIterators.remove(index);
                }
                knn = new Knn();
                knn.setOptions(parameters);
                knn.setSampleSize(1);
                knns.add(knn);
            } else {
                int index = random.nextInt(knns.size());
                knn = knns.get(index);
                int sampleSize = knn.getSampleSize() + 1;
                knn.setSampleSize(sampleSize);
                if(sampleSize + 1 > trainInstances.size()) {
                    knns.remove(index);
                }
            }
            knn = knn.copy();
            knn.buildClassifier(trainInstances);
            Candidate candidate = new Candidate(knn, knn.getTrainResults());
            selector.add(candidate);
            phaseTime = Long.max(System.nanoTime() - startPhaseTime, phaseTime);
            remainingParameters = !parameterSetIterators.isEmpty();
            remainingKnns = !knns.isEmpty();
            incrementTrainTimeNanos(System.nanoTime() - startTime);
        }
        List<Candidate> constituents = selector.getSelected();
        modules = new EnsembleModule[constituents.size()];
        for(int i = 0; i < constituents.size(); i++) {
            Knn knn = constituents.get(i).getKnn();
            modules[i] = new EnsembleModule(knn.toString(), knn, knn.getParameters());
            modules[i].trainResults = knn.getTrainResults();
        }
        weightingScheme.defineWeightings(modules, trainInstances.numClasses());
        votingScheme.trainVotingScheme(modules, trainInstances.numClasses());
        ClassifierResults trainResults = new ClassifierResults();
        setTrainResults(trainResults);
        for(int i = 0; i < trainInstances.size(); i++) {
            long predictionTime = System.nanoTime();
            double[] distribution = votingScheme.distributionForTrainInstance(modules, i);
            predictionTime = System.nanoTime() - predictionTime;
            for(int j = 0; j < modules.length; j++) {
                predictionTime += modules[j].trainResults.getPredictionTimeInNanos(i);
            }
            trainResults.addPrediction(trainInstances.get(i).classValue(), distribution, Utilities.argMax(distribution, getTrainRandom()), predictionTime, null);
        }
        incrementTrainTimeNanos(System.nanoTime() - startTime);
        setClassifierResultsMetaInfo(trainResults);
    }

    @Override
    public double[] distributionForInstance(final Instance testInstance) throws
                                                                     Exception {
        return votingScheme.distributionForInstance(modules, testInstance);
    }

    private class Candidate {
        private final Knn knn;
        private final ClassifierResults trainResults;

        private Candidate(final Knn knn, final ClassifierResults trainResults) {
            this.knn = knn;
            this.trainResults = trainResults;
        }

        public Knn getKnn() {
            return knn;
        }

        public ClassifierResults getTrainResults() {
            return trainResults;
        }

    }
}

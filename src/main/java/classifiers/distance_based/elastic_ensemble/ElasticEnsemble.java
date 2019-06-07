package classifiers.distance_based.elastic_ensemble;

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
import utilities.ArrayUtilities;
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
    private final List<Constituent> constituents = new ArrayList<>();
    private EnsembleModule[] modules = null;
    private ModuleWeightingScheme weightingScheme = new TrainAcc();
    private ModuleVotingScheme votingScheme = new MajorityVoteByConfidence();
    private long phaseTime = 0;
    private ClassifierResults trainResults = null;

    @Override
    public void buildClassifier(final Instances trainInstances) throws
                                                      Exception {
        long startTime = System.nanoTime();
        constituents.clear();
        parameterSpaces.clear();
        parameterSpaces.addAll(getParameterSpaces(trainInstances, parameterSpaceGetters));
        if(removeDuplicateParameterValues) {
            for(ParameterSpace parameterSpace : parameterSpaces) {
                parameterSpace.removeDuplicateValues();
            }
        }
        for(ParameterSpace parameterSpace : parameterSpaces) {
            Knn knn = new Knn();
            constituents.add(new Constituent(knn, new ParameterSetIterator(parameterSpace, new RandomIndexIterator(getTrainRandom(), parameterSpace.size())), trainInstances.size()));
        }
        incrementTrainTimeNanos(System.nanoTime() - startTime);
        while (!constituents.isEmpty() && remainingTrainContract() > phaseTime) {
            int constituentIndex = getTrainRandom().nextInt(constituents.size());
            Constituent constituent = constituents.get(constituentIndex);
            if(!constituent.hasNext()) {
                constituents.remove(constituentIndex);
            } else {
                long startPhaseTime = System.nanoTime();
                Knn knn = constituent.next();
                knn.buildClassifier(trainInstances);
                phaseTime = Long.max(System.nanoTime() - startPhaseTime, phaseTime);
            }
            incrementTrainTimeNanos(System.nanoTime() - startTime);
        }
        modules = new EnsembleModule[constituents.size()];
        for(int i = 0; i < constituents.size(); i++) {
            Knn knn = constituents.get(i).getKnn();
            modules[i] = new EnsembleModule(knn.toString(), knn, knn.getParameters());
            modules[i].trainResults = knn.getTrainResults();
        }
        weightingScheme.defineWeightings(modules, trainInstances.numClasses());
        votingScheme.trainVotingScheme(modules, trainInstances.numClasses());
        trainResults = new ClassifierResults();
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

    @Override
    public ClassifierResults getTrainResults() {
        return trainResults;
    }

    private class Constituent implements Iterator<Knn> {

        public Knn getKnn() {
            return knn;
        }

        private final Knn knn;
        private final Iterator<String[]> parameterIterator;
        private final int numTrainInstances;

        private Constituent(final Knn knn,
                            final Iterator<String[]> parameterIterator,
                            final int numTrainInstances) {
            this.knn = knn;
            knn.setSampleSize(0);
            this.parameterIterator = parameterIterator;
            this.numTrainInstances = numTrainInstances;
        }

        public boolean neighboursRemaining() {
            return knn.getSampleSize() < numTrainInstances;
        }

        public boolean parametersRemaining() {
            return parameterIterator.hasNext();
        }

        @Override
        public boolean hasNext() {
            return neighboursRemaining() || parametersRemaining();
        }

        @Override
        public Knn next() {
            // todo dimension priority
            if(neighboursRemaining()) {
                knn.setSampleSize(knn.getSampleSize() + 1);
            } else { // params remaining
                String[] params = parameterIterator.next();
                try {
                    knn.setOptions(params);
                } catch (Exception e) {
                    throw new IllegalStateException(e);
                }
            }
            return knn;
        }
    }
}

package classifiers.tuning;

import classifiers.distance_based.elastic_ensemble.iteration.AbstractIterator;
import classifiers.distance_based.elastic_ensemble.iteration.ParameterSetIterator;
import classifiers.distance_based.elastic_ensemble.iteration.limited.LimitedIterator;
import classifiers.distance_based.elastic_ensemble.iteration.random.RandomIterator;
import classifiers.distance_based.knn.Knn;
import classifiers.template.config.TemplateConfig;
import evaluation.storage.ClassifierResults;
import evaluation.tuning.ParameterSet;
import evaluation.tuning.ParameterSpace;
import utilities.ArrayUtilities;
import weka.classifiers.AbstractClassifier;
import weka.core.Instances;

import java.util.Comparator;
import java.util.function.BiFunction;
import java.util.function.Function;

public class TunedConfig extends TemplateConfig {

//    private Function<Instances, AbstractIterator<AbstractClassifier>> classifierIteratorGetter;
//    private AbstractIterator<AbstractClassifier> classifierIterator;
    private Comparator<ClassifierResults> comparator = Comparator.comparingDouble(ClassifierResults::getAcc);
    private final static String LIMIT_KEY = "il";
    private int limit = -1;
    private ParameterSet classifierParameterSet = new ParameterSet();
    private final static String LIMIT_PERCENTAGE_KEY = "ilp";
    private double limitPercentage = -1;
    private ParameterSpace parameterSpace;
    private Function<Instances, ParameterSpace> parameterSpaceGetter;
    private AbstractIterator<Integer> parameterSetIndexIterator;
    private final static String ITERATION_STRATEGY_KEY = "is";
    private IterationStrategy iterationStrategy = IterationStrategy.RANDOM;
    private ParameterSetIterator parameterSetIterator = new ParameterSetIterator();
    private RandomIterator<Integer> indexIterator = new RandomIterator<>();
    private LimitedIterator<ParameterSet> limitedIterator = new LimitedIterator<>();
    private Long seed = null;

    public ParameterSet getClassifierParameterSet() {
        return classifierParameterSet;
    }

    public void setClassifierParameterSet(ParameterSet classifierParameterSet) {
        this.classifierParameterSet = classifierParameterSet;
    }

    public void setLimit(int limit) {
        this.limit = limit;
    }

    public int getLimit() {
        return limit;
    }


    @Override
    public void setOptions(final String[] options) throws
                                                   Exception {
        ArrayUtilities.forEachPair(options, (key, value) -> {
            switch (key) {
                case LIMIT_KEY:
                    setLimit(Integer.parseInt(value));
                    break;
                case ITERATION_STRATEGY_KEY:
                    setIterationStrategy(IterationStrategy.fromString(value));
                    break;
                case LIMIT_PERCENTAGE_KEY:
                    setLimitPercentage(Double.parseDouble(value));
                    break;
            }
            return null;
        });
        if(classifierParameterSet != null) {
            classifierParameterSet.setOptions(options);
        }
    }

    @Override
    public String[] getOptions() {
        String[] options = new String[] {LIMIT_KEY,
                                         String.valueOf(limit),
                                         LIMIT_PERCENTAGE_KEY,
                                         String.valueOf(limitPercentage),
                                         ITERATION_STRATEGY_KEY,
                                         String.valueOf(iterationStrategy)
        };
        if(classifierParameterSet != null) {
            return ArrayUtilities.concat(classifierParameterSet.getOptions(), options);
        } else {
            return options;
        }
    }
    // todo keep more than 1, perhaps they should then be ensembled?

    public TunedConfig() {

    }

    public TunedConfig(TunedConfig other ){
        throw new UnsupportedOperationException();
    }

    @Override
    public TemplateConfig copy() throws Exception {
        throw new UnsupportedOperationException(); // todo
    }

    @Override
    public void copyFrom(Object object) throws Exception {
        throw new UnsupportedOperationException(); // todo
    }

    private ClassifierIterator classifierIterator = new ClassifierIterator();

    private boolean fullSetup = true;

    public void buildClassifierIterator(Instances trainSet) {
//        if(classifierIteratorGetter != null) {
//            classifierIterator = classifierIteratorGetter.apply(trainSet);
//        } else {
//        }
        if(fullSetup) {
            fullSetup = false;
            if(seed == null) {
                throw new UnsupportedOperationException();
            }
            if(parameterSpaceGetter != null) {
                parameterSpace = parameterSpaceGetter.apply(trainSet);
            }
            if(parameterSpace == null) {
                throw new IllegalStateException();
            }
            parameterSpace.removeDuplicateParameterSets();
            indexIterator.setSeed(seed);
            parameterSetIterator.setParameterSpace(parameterSpace);
            parameterSetIterator.setIterator(indexIterator);
            limitedIterator.setIterator(parameterSetIterator);
            classifierIterator.setIterator(limitedIterator);
            classifierIterator.setSupplier(Knn::new);
            indexIterator.addAll(ArrayUtilities.sequence(parameterSpace.size()));
        }
        int size = (int) (parameterSpace.size() * limitPercentage);
        limitedIterator.setLimit(size);
    }

    public Comparator<ClassifierResults> getComparator() {
        return comparator;
    }

    public void setComparator(Comparator<ClassifierResults> comparator) {
        this.comparator = comparator;
    }

//    public AbstractIterator<AbstractClassifier> getClassifierIterator() {
//        return classifierIterator;
//    }
//
//    public void setClassifierIterator(AbstractIterator<AbstractClassifier> classifierIterator) {
//        this.classifierIterator = classifierIterator;
//    }

    public Function<Instances, ParameterSpace> getParameterSpaceGetter() {
        return parameterSpaceGetter;
    }

    public void setParameterSpaceGetter(final Function<Instances, ParameterSpace> parameterSpaceGetter) {
        this.parameterSpaceGetter = parameterSpaceGetter;
    }

    public ParameterSpace getParameterSpace() {
        return parameterSpace;
    }

    public void setParameterSpace(final ParameterSpace parameterSpace) {
        this.parameterSpace = parameterSpace;
    }

    public AbstractIterator<Integer> getParameterSetIndexIterator() {
        return parameterSetIndexIterator;
    }

    public void setParameterSetIndexIterator(AbstractIterator<Integer> parameterSetIndexIterator) {
        this.parameterSetIndexIterator = parameterSetIndexIterator;
    }

    public IterationStrategy getIterationStrategy() {
        return iterationStrategy;
    }

    public void setIterationStrategy(IterationStrategy iterationStrategy) {
        this.iterationStrategy = iterationStrategy;
    }

    public double getLimitPercentage() {
        return limitPercentage;
    }

    public void setLimitPercentage(double limitPercentage) {
        this.limitPercentage = limitPercentage;
    }

    public Long getSeed() {
        return seed;
    }

    public void setSeed(Long seed) {
        this.seed = seed;
    }

    public AbstractIterator<AbstractClassifier> getClassifierIterator() {
        return classifierIterator;
    }
}

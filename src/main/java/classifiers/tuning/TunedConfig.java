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
import weka.classifiers.AbstractClassifier;
import weka.core.Instances;

import java.util.Comparator;
import java.util.function.Function;

public class TunedConfig extends TemplateConfig {

//    private Function<Instances, AbstractIterator<AbstractClassifier>> classifierIteratorGetter;
//    private AbstractIterator<AbstractClassifier> classifierIterator;
    private Comparator<ClassifierResults> comparator = Comparator.comparingDouble(ClassifierResults::getAcc);
    private final static String LIMIT_KEY = "il";
    private int limit = -1;
    private final static String LIMIT_PERCENTAGE_KEY = "ilp";
    private double limitPercentage = -1;
    private ParameterSpace parameterSpace;
    private Function<Instances, ParameterSpace> parameterSpaceGetter;
    private AbstractIterator<Integer> parameterSetIndexIterator;
    private final static String ITERATION_STRATEGY_KEY = "is";
    private IterationStrategy iterationStrategy = IterationStrategy.RANDOM;
    private Long seed = null;

    public void setLimit(int limit) {
        this.limit = limit;
    }

    public int getLimit() {
        return limit;
    }

    @Override
    public void setOption(final String key, final String value) {
        if(key.equals(LIMIT_KEY)) setLimit(Integer.parseInt(value));
        else if(key.equals(ITERATION_STRATEGY_KEY)) setIterationStrategy(IterationStrategy.fromString(value));
        else if(key.equals(LIMIT_PERCENTAGE_KEY)) setLimitPercentage(Double.parseDouble(value));
    }

    @Override
    public String[] getOptions() {
        return new String[] {LIMIT_KEY,
                             String.valueOf(limit),
                LIMIT_PERCENTAGE_KEY,
                String.valueOf(limitPercentage),
                ITERATION_STRATEGY_KEY,
                String.valueOf(iterationStrategy)
        };
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

    private AbstractIterator<AbstractClassifier> classifierIterator;

    public void buildClassifierIterator(Instances trainSet) {
//        if(classifierIteratorGetter != null) {
//            classifierIterator = classifierIteratorGetter.apply(trainSet);
//        } else {
//        }
        if(seed == null) {
            throw new UnsupportedOperationException();
        }
        if(parameterSpaceGetter != null) {
            parameterSpace = parameterSpaceGetter.apply(trainSet);
        }
        parameterSpace.removeDuplicateParameterSets();
        int size = (int) (parameterSpace.size() * limitPercentage);
        AbstractIterator<Integer> indexIterator = new RandomIterator<>(seed);
        AbstractIterator<ParameterSet> parameterSetIterator = new ParameterSetIterator(parameterSpace, indexIterator);
        parameterSetIterator = new LimitedIterator<>(parameterSetIterator, size);
        classifierIterator = new ClassifierIterator(Knn::new, parameterSetIterator);
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

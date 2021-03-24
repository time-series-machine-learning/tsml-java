package tsml.classifiers.distance_based.utils.stats.scoring;

import tsml.classifiers.distance_based.utils.collections.lists.RepeatList;
import utilities.ArrayUtilities;

import java.io.Serializable;
import java.util.*;

import static org.junit.Assert.assertEquals;
import static utilities.ArrayUtilities.normalise;


public class Labels<A> extends AbstractList<Label<A>> implements Serializable {

    public Labels(final List<A> labels, final List<Double> weights) {
        this();
        assertEquals(labels.size(), weights.size());
        for(int i = 0; i < labels.size(); i++) {
            add(new Label<>(labels.get(i), weights.get(i)));
        }
    }

    public Labels(final List<A> labels) {
        this(labels, new RepeatList<>(1d, labels.size()));
    }

    public Labels() {
        clear();
    }

    public Labels(Labels<A> labels) {
        this();
        addAll(labels);
    }

    private List<Label<A>> labels = new ArrayList<>();
    // label set is a list of unique labels. This is required specifically for labels which are not represented, i.e. given the labels 0,0,1,1,2,2,5,5. The labels are 1,2,3 and 5. Note there is no label 4. However, there is a label 3 which is not represented at all, and appears to not be an option like label 4. This leads to incorrect entropy / scoring calculations as the number of possible partitions is incorrectly reduced. I.e. given labels 0,2 a gini score would give 0.5 for two classes: 0 and 2. If however, there is in fact 3 classes, 0, 1 and 2, the gini score will be different. Therefore this label set is designed to specify the available labels, some of which may not be present in the main labels list whatsoever. 
    private List<A> labelSet;
    private TreeMap<A, Double> countsMap;
    private List<Double> distribution;
    private List<Double> countsList;
    private Double weightSum;

    @Override public String toString() {
        return labels.toString();
    }

    @Override public Label<A> set(final int i, final Label<A> label) {
        final Label<A> prev = labels.set(i, label);
        countsMap.compute(label.getId(), (a, count) -> {
            if(count == null) {
                throw new IllegalStateException("expected a count for previous");
            } else {
                return count - prev.getWeight() + label.getWeight();
            }
        });
        resetMetaData();
        return prev;
    }

    @Override public void add(final int i, final Label<A> label) {
        labels.add(i, label);
        countsMap.compute(label.getId(), (a, count) -> {
            if(count == null) {
                labelSet.add(label.getId());
                return label.getWeight();
            } else {
                return count + label.getWeight();
            }
        });
        resetMetaData();
    }

    @Override public Label<A> remove(final int i) {
        final Label<A> label = labels.remove(i);
        countsMap.compute(label.getId(), (a, count) -> {
            if(count == null) {
                throw new IllegalStateException("expected a count for previous");
            } else {
                return count - label.getWeight();
            }
        });
        resetMetaData();
        return label;
    }

    @Override public int size() {
        return labels.size();
    }

    @Override public void clear() {
        labels = new ArrayList<>();
        countsMap = new TreeMap<>();
        labelSet = new ArrayList<>();
        resetMetaData();
    }

    @Override public Label<A> get(final int i) {
        return labels.get(i);
    }

    private void resetMetaData() {
        setDistribution(null);
        setWeightSum(null);
    }

    public Labels<A> setLabels(final List<A> labels) {
        for(int i = 0; i < labels.size(); i++) {
            final double weight;
            if(i == size()) {
                weight = 1d;
            } else {
                weight = get(i).getWeight();
            }
            set(i, new Label<>(get(i).getId(), weight));
        }
        return this;
    }

    public Labels<A> setWeights(final List<Double> weights) {
        for(int i = 0; i < weights.size(); i++) {
            set(i, new Label<>(get(i).getId(), weights.get(i)));
        }
        return this;
    }

    public Labels<A> setDistribution(final List<Double> distribution) {
        this.distribution = distribution;
        return this;
    }

    public List<A> getLabelSet() {
        return Collections.unmodifiableList(labelSet);
    }

    public TreeMap<A, Double> getCountsMap() {
        return countsMap;
    }

    public List<Double> getDistribution() {
        if(distribution == null) {
            distribution = normalise(getCountsMap().values());
        }
        return distribution;
    }

    public Map<A, Double> getDistributionMap() {
        return new AbstractMap<A, Double>() {
            @Override public Set<Entry<A, Double>> entrySet() {
                return getCountsMap().entrySet();
            }

            @Override public Double get(final Object o) {
                Double value = getCountsMap().get(o);
                if(value != null) {
                    value /= getWeightSum();
                }
                return value;
            }
        };
    }

    public Labels<A> setLabelSet(final List<A> labelSet) {
        this.labelSet = labelSet;
        if(labelSet != null) {
            for(A label : labelSet) {
                countsMap.computeIfAbsent(label, x -> 0d);
            }
            final HashSet<A> set = new HashSet<>(labelSet);
            for(A label : countsMap.keySet()) {
                if(!set.contains(label)) {
                    throw new IllegalArgumentException("label set " + labelSet + " does not contain the label " + label);
                }
            }
        }
        return this;
    }

    public List<A> getLabels() {
        return new AbstractList<A>() {
            @Override public A get(final int i) {
                return Labels.this.get(i).getId();
            }

            @Override public int size() {
                return Labels.this.size();
            }
        };
    }

    public List<Double> getWeights() {
        return new AbstractList<Double>() {
            @Override public Double get(final int i) {
                return Labels.this.get(i).getWeight();
            }

            @Override public int size() {
                return Labels.this.size();
            }
        };
    }

    public List<Double> getCountsList() {
        if(countsList == null) {
            final Map<A, Double> countsMap = getCountsMap();
            countsList = Collections.unmodifiableList(new ArrayList<>(countsMap.values()));
        }
        return countsList;
    }

    protected static Labels<Integer> fromCounts(List<Double> countsList) {
        final Labels<Integer> labels = new Labels<>();
        labels.setLabelSet(ArrayUtilities.sequence(countsList.size()));
        TreeMap<Integer, Double> map = new TreeMap<>();
        double sum = 0;
        for(int i = 0; i < countsList.size(); i++) {
            final Double count = countsList.get(i);
            map.put(i, count);
            sum += count;
        }
        labels.setCountsMap(map);
        labels.setWeightSum(sum);
        return labels;
    }

    public Labels<A> setCountsMap(final TreeMap<A, Double> countsMap) {
        this.countsMap = Objects.requireNonNull(countsMap);
        return this;
    }

    public double getWeightSum() {
        if(weightSum == null) {
            weightSum = stream().mapToDouble(Label::getWeight).sum();
        }
        return weightSum;
    }

    public Labels<A> setWeightSum(final Double weightSum) {
        this.weightSum = weightSum;
        return this;
    }
}

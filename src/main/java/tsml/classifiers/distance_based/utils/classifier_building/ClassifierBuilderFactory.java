package tsml.classifiers.distance_based.utils.classifier_building;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.distance_based.ee.EeConfig;
import tsml.classifiers.distance_based.knn.configs.KNNConfig;
import weka.classifiers.Classifier;

import java.util.*;
import java.util.function.Supplier;
import java.util.stream.Collectors;

public class ClassifierBuilderFactory {

    public interface Tag {
        String getName();
    }

    public interface ClassifierBuilder {
        String toString();
        String getName();
        Classifier build();
        ImmutableList<? extends Tag> getTags();
    }

    public static class SuppliedClassifierBuilder implements ClassifierBuilder {
        private final String name;
        private final ImmutableList<? extends Tag> tags;
        private final Supplier<? extends Classifier> supplier;

        public SuppliedClassifierBuilder(final String name,
                                         final Supplier<? extends Classifier> supplier,
                                         final Tag... tags) {
            this(name, supplier, Arrays.asList(tags));
        }

        public SuppliedClassifierBuilder(final String name, final Supplier<? extends Classifier> supplier,
                                         final List<? extends Tag> tags) { // todo i'd like this to be an
            // Iterable<Supplier<String>> rather than list but somehow it conflicts with the other constructor!! Odd.
            this.name = name;
            this.supplier = supplier;
            this.tags = ImmutableList.copyOf(tags);
        }

        public String getName() {
            return name;
        }

        public ImmutableList<? extends Tag> getTags() {
            return tags;
        }

        public Supplier<? extends Classifier> getSupplier() {
            return supplier;
        }

        public Classifier build() {
            Classifier classifier = getSupplier().get();
            if(classifier instanceof EnhancedAbstractClassifier) {
                ((EnhancedAbstractClassifier) classifier).setClassifierName(getName());
            }
            // todo we could also set tags then the classifier knows it's capabilities...
            return classifier;
        }

        @Override public String toString() {
            return name;
        }
    }

    private static ClassifierBuilderFactory INSTANCE = new ClassifierBuilderFactory();
    private final Map<String, ClassifierBuilder> classifierBuildersByName = new TreeMap<>();
    private final Map<Tag, Set<ClassifierBuilder>> classifierBuildersByTag = new TreeMap<>((tag, t1) -> tag.getName().compareToIgnoreCase(t1.getName()));
    private final Set<ClassifierBuilder> classifierBuilders = new HashSet<>();

    public ClassifierBuilderFactory() {}

    @Override public String toString() {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append(getClass().getSimpleName()).append("{").append(System.lineSeparator());
        for(Map.Entry<String, ClassifierBuilder> entry : classifierBuildersByName.entrySet()) {
            stringBuilder.append("\t");
            stringBuilder.append(entry.getKey());
            stringBuilder.append(": ");
            stringBuilder.append(entry.getValue().getTags().toString());
            stringBuilder.append(System.lineSeparator());
        }
        stringBuilder.append("}");
        return stringBuilder.toString();
    }

    public static ClassifierBuilderFactory getGlobalInstance() {
        return INSTANCE;
    }

    public void addAll(List<ClassifierBuilder> all) {
        for(ClassifierBuilder builder : all) {
            add(builder);
        }
    }

    public void addAll(ClassifierBuilder... builders) {
        addAll(Arrays.asList(builders));
    }

    public void addAll(Supplier<ClassifierBuilder>... suppliers) {
        addAll(Arrays.stream(suppliers).map(Supplier::get).collect(Collectors.toList()));
    }

    public void add(ClassifierBuilder classifierBuilder) {
        String name = classifierBuilder.getName();
        name = name.toLowerCase();
        if(classifierBuildersByName.containsKey(name)) {
            throw new IllegalArgumentException("oops, a classifier already exists under the name: " + name);
        } else if(classifierBuilders.contains(classifierBuilder)) {
            throw new IllegalArgumentException("oops, a classifier already exists under that supplier.");
        } else {
            classifierBuilders.add(classifierBuilder);
            classifierBuildersByName.put(name, classifierBuilder);
        }
        for(Tag tag : classifierBuilder.getTags()) {
            classifierBuildersByTag.computeIfAbsent(tag, k -> new HashSet<>()).add(classifierBuilder);
        }
    }

    public ClassifierBuilder getClassifierBuilderByName(String name) {
        name = name.toLowerCase();
        ClassifierBuilder classifierBuilder = classifierBuildersByName.get(name);
        return classifierBuilder;
    }

    public Set<ClassifierBuilder> getClassifierBuildersByNames(String... names) {
        return getClassifierBuildersByNames(Arrays.asList(names));
    }

    public Set<ClassifierBuilder> getClassifierBuildersByNames(Iterable<String> names) {
        Set<ClassifierBuilder> set = new HashSet<>();
        for(String name : names) {
            set.add(getClassifierBuilderByName(name));
        }
        return ImmutableSet.copyOf(set);
    }

    public Set<ClassifierBuilder> getClassifierBuildersByTag(String tag) {
        tag = tag.toLowerCase();
        Set<ClassifierBuilder> classifierBuilders = classifierBuildersByTag.get(tag);
        if(classifierBuilders != null) {
            return ImmutableSet.copyOf(classifierBuilders);
        } else {
            return ImmutableSet.of();
        }
    }

    public Set<ClassifierBuilder> getClassifierBuildersByTags(Iterable<String> tags) {
        Set<ClassifierBuilder> set = new HashSet<>();
        for(String tag : tags) {
            set.addAll(getClassifierBuildersByTag(tag));
        }
        return ImmutableSet.copyOf(set);
    }

    public Set<ClassifierBuilder> getClassifierBuildersByTags(String... tags) {
        return getClassifierBuildersByTags(Arrays.asList(tags));
    }

    static {
        getGlobalInstance().addAll(KNNConfig.all());
        getGlobalInstance().addAll(EeConfig.all());
    }

    public static void main(String[] args) {
        System.out.println(getGlobalInstance().toString());
    }
}

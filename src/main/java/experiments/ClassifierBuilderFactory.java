package experiments;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.distance_based.knn.KnnConfigs;
import weka.classifiers.Classifier;

import java.util.*;
import java.util.function.Supplier;

public class ClassifierBuilderFactory {

    public static class ClassifierBuilder<A extends Classifier> {
        private final String name;
        private final ImmutableList<String> tags;
        private final Supplier<A> supplier;

        public ClassifierBuilder(final String name,
                                 final Supplier<A> supplier,
                                 final String... tags) {
            this(name, supplier, Arrays.asList(tags));
        }

        public ClassifierBuilder(final String name, final Supplier<A> supplier, final Iterable<String> tags) {
            this.name = name;
            this.tags = ImmutableList.copyOf(tags);
            this.supplier = supplier;
        }

        public String getName() {
            return name;
        }

        public ImmutableList<String> getTags() {
            return tags;
        }

        public Supplier<A> getSupplier() {
            return supplier;
        }

        public A build() {
            A classifier = getSupplier().get();
            if(classifier instanceof EnhancedAbstractClassifier) {
                ((EnhancedAbstractClassifier) classifier).setClassifierName(getName());
            }
            // todo we could also set tags then the classifier knows it's capabilities...
            return classifier;
        }
    }

    private static ClassifierBuilderFactory INSTANCE = new ClassifierBuilderFactory();
    private final Map<String, ClassifierBuilder<?>> classifiersByName = new HashMap<>();
    private final Map<String, Set<ClassifierBuilder<?>>> classifierByTag = new HashMap<>();

    public ClassifierBuilderFactory() {}

    public static ClassifierBuilderFactory getInstance() {
        return INSTANCE;
    }

    public void add(ClassifierBuilder<?> classifierBuilder) {
        String name = classifierBuilder.getName();
        name = name.toLowerCase();
        ClassifierBuilder<?> current = classifiersByName.get(name);
        if(current != null) {
            throw new IllegalArgumentException("oops, a classifier already exists under the name: " + name);
        } else {
            classifiersByName.put(name, classifierBuilder);
        }
        for(String tag : classifierBuilder.getTags()) {
            classifierByTag.computeIfAbsent(tag, k -> new HashSet<>()).add(classifierBuilder);
        }
    }

    public static void addGlobal(ClassifierBuilder<?> classifierBuilder) {
        getInstance().add(classifierBuilder);
    }

    public ClassifierBuilder<?> getClassifierBuilderByName(String name) {
        name = name.toLowerCase();
        ClassifierBuilder<?> classifierBuilder = classifiersByName.get(name);
        if(classifierBuilder == null) {
            throw new IllegalArgumentException("oops, there's no classifier by the name: " + name);
        }
        return classifierBuilder;
    }

    public Set<ClassifierBuilder<?>> getClassifierBuildersByNames(String... names) {
        return getClassifierBuildersByNames(Arrays.asList(names));
    }

    public Set<ClassifierBuilder<?>> getClassifierBuildersByNames(Iterable<String> names) {
        Set<ClassifierBuilder<?>> set = new HashSet<>();
        for(String name : names) {
            set.add(getClassifierBuilderByName(name));
        }
        return ImmutableSet.copyOf(set);
    }

    public static ClassifierBuilder<?> getClassifierBuilderByNameGlobal(String name) {
        return getInstance().getClassifierBuilderByName(name);
    }

    public static Set<ClassifierBuilder<?>> getClassifierBuildersByNamesGlobal(String... names) {
        return getInstance().getClassifierBuildersByNames(names);
    }

    public static Set<ClassifierBuilder<?>> getClassifierBuildersByNamesGlobal(Iterable<String> names) {
        return getInstance().getClassifierBuildersByNames(names);
    }

    public Set<ClassifierBuilder<?>> getClassifierBuildersByTag(String tag) {
        tag = tag.toLowerCase();
        Set<ClassifierBuilder<?>> classifierBuilders = classifierByTag.get(tag);
        if(classifierBuilders != null) {
            return ImmutableSet.copyOf(classifierBuilders);
        } else {
            return ImmutableSet.of();
        }
    }

    public Set<ClassifierBuilder<?>> getClassifierBuildersByTags(Iterable<String> tags) {
        Set<ClassifierBuilder<?>> set = new HashSet<>();
        for(String tag : tags) {
            set.addAll(getClassifierBuildersByTag(tag));
        }
        return ImmutableSet.copyOf(set);
    }

    public Set<ClassifierBuilder<?>> getClassifierBuildersByTags(String... tags) {
        return getClassifierBuildersByTags(Arrays.asList(tags));
    }

    public static Set<ClassifierBuilder<?>> getClassifierBuildersByTagGlobal(String tag) {
        return getInstance().getClassifierBuildersByTag(tag);
    }

    public static Set<ClassifierBuilder<?>> getClassifierBuildersByTagsGlobal(Iterable<String> tags) {
        return getInstance().getClassifierBuildersByTags(tags);
    }

    public static Set<ClassifierBuilder<?>> getClassifierBuildersByTagsGlobal(String... tags) {
        return getInstance().getClassifierBuildersByTags(tags);
    }

    static {
//        addGlobal(new ClassifierBuilder<EnhancedAbstractClassifier>("DTWCV", KnnConfigs::buildDtw1nnV1, "similarity"));
    }
}

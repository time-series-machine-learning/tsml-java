package timeseriesweka.classifiers.distance_based.ee.selection;

import java.util.*;
import java.util.function.Function;

public class KBestPerTypeSelector<A, B> implements Selector<A> {

    protected final Map<B, KBestSelector<A, ?>> map = new HashMap<>();

    public <C extends Comparable<C>> void addSelector(KBestSelector<A, C> selector, B type) {
        map.put(type, selector);
    }

    private Function<B, KBestSelector<A, ?>> generator;

    public Function<B, KBestSelector<A, ?>> getGenerator() {
        return generator;
    }

    public void setGenerator(final Function<B, KBestSelector<A, ?>> generator) {
        this.generator = generator;
    }

    protected Function<A, B> extractor;

    public KBestPerTypeSelector() {}

    @Override
    public void add(A candidate) {
        if(extractor == null) {
            throw new IllegalStateException("need to set type extractor!");
        }
        B type = extractor.apply(candidate);
        KBestSelector<A, ?> kBestSelector = map.computeIfAbsent(type, generator);
        kBestSelector.add(candidate);
    }

    @Override
    public List<A> getSelected() {
        List<A> selected = new ArrayList<>();
        for(KBestSelector<A, ?> kBestSelector : map.values()) {
            selected.addAll(kBestSelector.getSelectedAsListWithDraws());
        }
        return selected;
    }

    public Function<A, B> getExtractor() {
        return extractor;
    }

    public void setExtractor(Function<A, B> extractor) {
        this.extractor = extractor;
    }

    @Override
    public void clear() {
        map.clear();
    }

    @Override
    public Object shallowCopy() throws
                         Exception {
        throw new UnsupportedOperationException();
    }

    @Override
    public void shallowCopyFrom(final Object object) throws
                                              Exception {
        throw new UnsupportedOperationException();
    }
}

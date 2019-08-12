package timeseriesweka.classifiers.distance_based.ee.selection;

import java.util.*;
import java.util.function.Function;

public class KBestPerTypeSelector<A, B> implements Selector<A> {

    private Map<B, KBestSelector<A, ?>> map = new HashMap<>();

    public <C extends Comparable<C>> void addSelector(KBestSelector<A, C> selector, B type) {
        map.put(type, selector);
    }

    private Function<B, KBestSelector<A, ?>> selectorFunction = b -> new KBestSelector<>();
    private Function<A, B> typeExtractor;

    public KBestPerTypeSelector() {}

    public KBestPerTypeSelector(Function<A, B> typeExtractor) { setTypeExtractor(typeExtractor); }

    @Override
    public void add(A candidate) {
        if(typeExtractor == null) {
            throw new IllegalStateException("need to set type extractor!");
        }
        B type = typeExtractor.apply(candidate);
        KBestSelector<A, ?> kBestSelector = map.computeIfAbsent(type, selectorFunction);
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

    public Function<A, B> getTypeExtractor() {
        return typeExtractor;
    }

    public void setTypeExtractor(Function<A, B> typeExtractor) {
        this.typeExtractor = typeExtractor;
    }

    @Override
    public void clear() {
        map.clear();
    }

    @Override
    public Object copy() throws
                         Exception {
        throw new UnsupportedOperationException();
    }

    @Override
    public void copyFrom(final Object object) throws
                                              Exception {
        throw new UnsupportedOperationException();
    }
}

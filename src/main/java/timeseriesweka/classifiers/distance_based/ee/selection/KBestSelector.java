package timeseriesweka.classifiers.distance_based.ee.selection;

import timeseriesweka.classifiers.distance_based.knn.Knn;
import utilities.ArrayUtilities;
import utilities.Copyable;

import java.util.*;
import java.util.function.Function;

public class KBestSelector<A, B> implements Copyable
 {
    private TreeMap<B, List<A>> map;
    private Function<A, B> extractor;
    private int size = 0;
    private int limit = -1;
    private B worstValue;
    private List<A> worstItems;
    private Comparator<B> comparator;

    public TreeMap<B, List<A>> getSelectedAsMapWithDraws() {
        return map;
    }

    public TreeMap<B, List<A>> getSelectedAsMap(Random random) { // todo move to util func
        TreeMap<B, List<A>> map = new TreeMap<>(getSelectedAsMapWithDraws());
        if(size > limit) {
            map = new TreeMap<>(map);
            Map.Entry<B, List<A>> lastEntry = map.lastEntry();
            List<A> list = new ArrayList<>(lastEntry.getValue());
            map.put(lastEntry.getKey(), list);
            int diff = size - limit;
            while (diff > 0) {
                diff--;
                list.remove(random.nextInt(list.size()));
            }
        }
        return map;
    }

     public List<A> getSelectedAsListWithDraws() {
         TreeMap<B, List<A>> map = getSelectedAsMapWithDraws();
         return ArrayUtilities.flatten(map);
     }

    public List<A> getSelectedAsList(Random random) {
        // todo sort this out
        TreeMap<B, List<A>> map = getSelectedAsMap(random);
        return ArrayUtilities.flatten(map);
    }

    @Override
    public void shallowCopyFrom(final Object object) throws
                                              Exception {
        KBestSelector<A, B> other = (KBestSelector<A, B>) object;
        setLimit(other.getLimit());
        setExtractor(other.getExtractor());
        size = other.size;
        map.clear();
        for(Map.Entry<B, List<A>> entry : other.map.entrySet()) {
            map.put(entry.getKey(), new ArrayList<>(entry.getValue()));
        }
        Map.Entry<B, List<A>> lastEntry = map.lastEntry();
        worstValue = lastEntry.getKey();
        worstItems = lastEntry.getValue();
    }

    public KBestSelector(Comparator<B> comparator) {
        map = new TreeMap<>(comparator.reversed());
        this.comparator = comparator;
    }

    public KBestSelector(KBestSelector<A, B> other) throws
                                            Exception {
        shallowCopyFrom(other);
    }

    public B getWorstValue() {
        return worstValue;
    }

    public void add(A item) {
        if(extractor == null) {
            throw new IllegalStateException("extractor not set");
        }
        B value = extractor.apply(item);
        int comparison;
        if(worstValue == null) {
            comparison = 1;
        } else {
            comparison = comparator.compare(value, worstValue);
        }
        if (comparison >= 0 || (size < limit || limit <= 0)) {
            List<A> items = map.get(value);
            if (items == null) {
                items = new ArrayList<>();
                map.put(value, items);
                if(size == 0) {
                    worstValue = value;
                    worstItems = items;
                } else if(comparison < 0){
                    worstValue = value;
                }
            }
            items.add(item);
            size++;
            if (comparison > 0 && size > limit && limit > 0) {
                int numFurthestItems = worstItems.size();
                if (size - limit >= numFurthestItems) {
                    map.pollLastEntry();
                    size -= numFurthestItems;
                    Map.Entry<B, List<A>> furthestNeighboursEntry = map.lastEntry();
                    worstItems = furthestNeighboursEntry.getValue();
                    worstValue = furthestNeighboursEntry.getKey();
                }
            }
        }
    }

    public void addAll(Collection<A> collection) {
        for(A item : collection) {
            add(item);
        }
    }

    public void clear() {
        map.clear();
        size = 0;
    }

    public int getLimit() {
        return limit;
    }

    public void setLimit(final int limit) {
        this.limit = limit;
    }

    public Function<A, B> getExtractor() {
        return extractor;
    }

    public void setExtractor(final Function<A, B> extractor) {
        this.extractor = extractor;
    }

    @Override
    public Object shallowCopy() throws
                         Exception {
        return new KBestSelector<>(this);
    }
 }

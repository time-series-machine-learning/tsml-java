package timeseriesweka.filters.cache;

public class DupeCache<A, B> extends Cache<A, A, B> {

    @Override
    public B get(final A firstKey, final A secondKey) {
        B result = super.get(firstKey, secondKey);
        if(result == null) {
            result = super.get(secondKey, firstKey);
        }
        return result;
    }

    @Override
    public void put(final A firstKey, final A secondkey, final B value) {
        super.remove(secondkey, firstKey);
        super.put(firstKey, secondkey, value);
    }

    @Override
    public void remove(final A firstKey, final A secondKey) {
        super.remove(firstKey, secondKey);
        super.remove(secondKey, firstKey);
    }
}

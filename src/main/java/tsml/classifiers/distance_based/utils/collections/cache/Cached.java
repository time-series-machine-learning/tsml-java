package tsml.classifiers.distance_based.utils.collections.cache;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */

public abstract class Cached {

    private boolean read = true;
    private boolean write = true;

    public boolean isWrite() {
        return write;
    }

    public Cached setWrite(final boolean write) {
        this.write = write;
        return this;
    }

    public boolean isRead() {
        return read;
    }

    public Cached setRead(final boolean read) {
        this.read = read;
        return this;
    }

    public boolean isReadOrWrite() {
        return isWrite() || isRead();
    }

    public boolean isReadAndWrite() {
        return isWrite() && isRead();
    }

    public boolean isReadOnly() {
        return isRead() && !isWrite();
    }

    public boolean isWriteOnly() {
        return !isRead() && isWrite();
    }

    public boolean isOff() {
        return !isRead() && !isWrite();
    }
}

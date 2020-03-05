package tsml.classifiers.distance_based.utils.stopwatch;

import java.util.HashSet;
import java.util.Set;

/**
 * Purpose: managed boolean state which can be suspended if required (i.e. ignored). Has listeners to pass on state
 * changes and corresponding checks for state changes, etc.
 *
 * Contributors: goastler
 */
public class Stated {

    public Stated() {}

    private transient Stated.State state = Stated.State.DISABLED;
    private transient Stated.State suspendedState = null;
    private transient Set<Stated> listeners = new HashSet<>();

    public void addListener(Stated stated) {
        listeners.add(stated);
    }

    public void removeListener(Stated stated) {
        listeners.remove(stated);
    }

    public Stated(final State state) {
        setStateAnyway(state);
    }

    public enum State {
        ENABLED,
        DISABLED,
    }

    public Stated suspend() {
        if(isUnsuspended()) {
            suspendedState = state;
            disableAnywayUnchecked();
            for(Stated listener : listeners) {
                listener.suspend();
            }
        }
        return this;
    }

    public boolean isSuspended() {
        return suspendedState != null;
    }

    public boolean isUnsuspended() {
        return !isSuspended();
    }

    public Stated unsuspend() {
        if(isSuspended()) {
            state = suspendedState;
            suspendedState = null;
            setStateAnyway(state);
            for(Stated listener : listeners) {
                listener.unsuspend();
            }
        }
        return this;
    }

    public Stated checkDisabled() {
        if(isSuspended()) {
            throw new IllegalStateException("already suspended");
        }
        if(!isDisabled()) {
            throw new IllegalStateException("not disabled");
        }
        return this;
    }

    public Stated checkEnabled() {
        if(isSuspended()) {
            throw new IllegalStateException("already suspended");
        }
        if(!isEnabled()) {
            throw new IllegalStateException("not enabled");
        }
        return this;
    }

    public Stated checkNotDisabled() {
        if(isSuspended()) {
            throw new IllegalStateException("already suspended");
        }
        if(isDisabled()) {
            throw new IllegalStateException("already disabled");
        }
        return this;
    }

    public Stated checkNotEnabled() {
        if(isSuspended()) {
            throw new IllegalStateException("already suspended");
        }
        if(isEnabled()) {
            throw new IllegalStateException("already enabled");
        }
        return this;
    }


    public Stated.State getState() {
        return state;
    }

    public Stated.State getSuspendedState() {
        return suspendedState;
    }

    public boolean isDisabled() {
        return getState().equals(Stated.State.DISABLED);
    }

    public boolean isEnabled() {
        return getState().equals(Stated.State.ENABLED);
    }

    public boolean setState(State state) {
        if(state.equals(Stated.State.DISABLED)) {
            return disable();
        } else if(state.equals(Stated.State.ENABLED)) {
            return enable();
        } else {
            throw new UnsupportedOperationException("invalid state");
        }
    }

    public boolean setStateAnyway(State state) {
        if(state.equals(Stated.State.DISABLED)) {
            return disableAnyway();
        } else if(state.equals(Stated.State.ENABLED)) {
            return enableAnyway();
        } else {
            throw new UnsupportedOperationException("invalid state");
        }
    }

    public boolean enable() {
        checkNotEnabled();
        return enableAnyway();
    }

    public boolean enableAnyway() {
        if(isSuspended()) {
            throw new IllegalStateException("suspended");
        }
        return enableAnywayUnchecked();
    }

    private boolean enableAnywayUnchecked() {
        if(!isEnabled()) {
            state = Stated.State.ENABLED;
            for(Stated listener : listeners) {
                listener.enableAnyway();
            }
            return true;
        }
        return false;
    }

    public boolean disable() {
        checkNotDisabled();
        return disableAnyway();
    }

    public boolean disableAnyway() {
        if(isSuspended()) {
            throw new IllegalStateException("suspended");
        }
        return disableAnywayUnchecked();
    }

    private boolean disableAnywayUnchecked() {
        if(!isDisabled()) {
            state = Stated.State.DISABLED;
            for(Stated listener : listeners) {
                listener.disableAnyway();
            }
            return true;
        }
        return false;
    }

    @Override public String toString() {
        return "Stated{" +
            "state=" + state +
            ", suspendedState=" + suspendedState +
            '}';
    }
}

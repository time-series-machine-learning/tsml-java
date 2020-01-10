package utilities;

public class Stated {

    public Stated() {}

    private Stated.State state = Stated.State.DISABLED;
    private Stated.State suspendedState = null;

    public Stated(final State state) {
        setStateAnyway(state);
    }

    public enum State {
        ENABLED,
        DISABLED,
    }

    public Stated suspend() {
        suspendedState = state;
        disableAnyway();
        return this;
    }

    public boolean isSuspended() {
        return suspendedState != null;
    }

    public Stated unsuspend() {
        state = suspendedState;
        suspendedState = null;
        setStateAnyway(state);
        return this;
    }

    public Stated checkDisabled() {
        if(!isDisabled()) {
            throw new IllegalStateException("not disabled");
        }
        return this;
    }

    public Stated checkEnabled() {
        if(!isEnabled()) {
            throw new IllegalStateException("not enabled");
        }
        return this;
    }

    public Stated checkNotDisabled() {
        if(isDisabled()) {
            throw new IllegalStateException("already disabled");
        }
        return this;
    }

    public Stated checkNotEnabled() {
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
        if(!isEnabled()) {
            state = Stated.State.ENABLED;
            return true;
        }
        return false;
    }

    public boolean disable() {
        checkNotDisabled();
        return disableAnyway();
    }

    public boolean disableAnyway() {
        if(!isDisabled()) {
            state = Stated.State.DISABLED;
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

package utilities;


public class Checks {
    private Checks() {

    }


    public static boolean isValidPercentage(double percentage) {
        return percentage >= 0 && percentage <= 1;
    }
}

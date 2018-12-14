public class App {
    public String getGreeting() {

        return "abc";
    }

    public static void main(String[] args) {
        System.out.println(new App().getGreeting());
    }
}
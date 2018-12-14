plugins {
    java
    application
}

repositories {
    mavenCentral()    
    jcenter() 
}

dependencies {
    testImplementation("junit:junit:4.12") 
}

application {
    mainClassName = "App"
}
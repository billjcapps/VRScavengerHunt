import unfiltered.jetty.SocketPortBinding

/**
  * Created by AC010168 on 2/20/2017.
  */
object SparkServer extends App {
  val bindingIP = SocketPortBinding(host = "192.168.1.177", port = 8080)
  unfiltered.jetty.Server.portBinding(bindingIP).plan(SparkPlan).run()
}

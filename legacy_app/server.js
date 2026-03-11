var http = require("http");
var url = require("url");

var customers = [
  { id: 1, name: "Alice", tier: "gold" },
  { id: 2, name: "Bob", tier: "silver" }
];

function findCustomer(id, callback) {
  setTimeout(function () {
    var customer = customers.filter(function (item) {
      return item.id === id;
    })[0];

    callback(customer || null);
  }, 5);
}

var server = http.createServer(function (req, res) {
  var parsed = url.parse(req.url, true);

  if (parsed.pathname === "/customers" && req.method === "GET") {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify(customers));
    return;
  }

  if (parsed.pathname.indexOf("/customers/") === 0 && req.method === "GET") {
    var id = parseInt(parsed.pathname.split("/")[2], 10);

    findCustomer(id, function (customer) {
      if (!customer) {
        res.writeHead(404, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "Customer not found" }));
        return;
      }

      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify(customer));
    });

    return;
  }

  res.writeHead(404, { "Content-Type": "application/json" });
  res.end(JSON.stringify({ error: "Route not found" }));
});

server.listen(3000, function () {
  console.log("Legacy server listening on port 3000");
});

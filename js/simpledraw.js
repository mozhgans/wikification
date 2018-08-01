function simpledraw(word_array, eid) {

  //a weird idempotency thing
  //$(eid).width("960px");
  //$(eid).height("600px");        
  //var margin = {top: 20, right: 20, bottom: 30, left: 40};
  //var width =  $(eid).width()- margin.left - margin.right;
  //var height = $(eid).height() - margin.top - margin.bottom;
  //var svg = d3.select(eid).append("svg")
  // .style("position", "relative")
  //  .style("max-width", "500px")
  //  .attr("width", width + "px")
  //  .attr("height", (height + 50) + "px")
  //  .append("g")
   // .attr("transform", "translate(" + margin.left + "," + margin.top + ")");    
    
var fill = d3.scale.category20();

var layout = d3.layout.cloud()
    .size([450, 500])
    .words(word_array)
    .padding(5)
    .rotate(function() { return 0; })
    .font("sans-serif")
    .fontSize(function(d) { return d.size; })
    .on("end", draw);

layout.start();

function draw(words) {
  d3.select(eid).append("svg")
      .attr("width", layout.size()[0])
      .attr("height", layout.size()[1])
    .append("g")
      .attr("transform", "translate(" + layout.size()[0] / 2 + "," + layout.size()[1] / 2 + ")")
    .selectAll("text")
      .data(words)
    .enter().append("text")
      .style("font-size", function(d) { return d.size + "px"; })
      .style("font-family", "sans-serif")
      .style("fill", function(d, i) { return fill(i); })
      .attr("text-anchor", "middle")
      .attr("transform", function(d) {
        return "translate(" + [d.x, d.y] + ")rotate(0)";
      })
      .text(function(d) { return d.text; });
}


};


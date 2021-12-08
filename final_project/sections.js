let dataset, svg, new_dataset
let salarySizeScale, salaryXScale, categoryColorScale
let simulation, nodes
let categoryLegend, salaryLegend

const categories = ['We survive on current income', 'We live comfortably on current income', 'We are struggling with current income', 'We are in a very critical situation with current income', 'No Response']

//
const categoriesXY = {'No Response': [0, 150, 42200, 48.3],
                        'We live comfortably on current income': [0, 450, 42745, 31.2],
                        'We survive on current income': [0, 850, 36900, 40.5],
                        'We are struggling with current income': [450, 450, 31913, 63.2],
                        'We are in a very critical situation with current income': [450, 850, 30100, 79.4],
                        'Not Asked': [-10000, -10000, 36900, 40.5],
                        'Increased': [-10000, -10000, 42745, 31.2],
                        'Diminished': [-10000, -10000, 30100, 79.4],
                        'Same / No change': [-10000, -10000, 31913, 63.2],
                        'No violent acts / crime':  [-10000, -10000, 36900, 40.5],
                        'Yes': [0, 450, 42200, 48.3],
                        'Not': [300, 450, 42745, 31.2]}

const categories2 = ['Yes', 'Not', 'No Response']

const categories2XY = {'Yes': [0, 450, 42200, 48.3],
                        'Not': [300, 450, 42745, 31.2],
                        'No Response': [600, 450, 36900, 40.5],
                        'We live comfortably on current income': [-10000, -10000, 42745, 31.2],
                        'We survive on current income': [-10000, -10000, 36900, 40.5],
                        'We are struggling with current income': [-10000, -10000, 31913, 63.2],
                        'We are in a very critical situation with current income': [-10000, -10000,  30100, 79.4],
                        'Not Asked': [-10000, -10000, 36900, 40.5],
                        'Increased': [-10000, -10000, 42745, 31.2],
                        'Diminished': [-10000, -10000, 30100, 79.4],
                        'Same / No change': [-10000, -10000, 31913, 63.2],
                        'No violent acts / crime':  [-10000, -10000, 36900, 40.5]}

const categories3 = ['Yes', 'Not', 'No Response', 'Not Asked']

const categories3XY = {'Yes': [0, 450, 42200, 48.3],
                        'Not': [300, 450, 42745, 31.2],
                        'No Response': [600, 450, 36900, 40.5],
                        'Not Asked': [300, 800, 36900, 40.5],
                        'We live comfortably on current income': [-10000, -10000, 42745, 31.2],
                        'We survive on current income': [-10000, -10000, 36900, 40.5],
                        'We are struggling with current income': [-10000, -10000, 31913, 63.2],
                        'We are in a very critical situation with current income': [-10000, -10000,  30100, 79.4],
                        'Increased': [-10000, -10000, 42745, 31.2],
                        'Diminished': [-10000, -10000, 30100, 79.4],
                        'Same / No change': [-10000, -10000, 31913, 63.2],
                        'No violent acts / crime':  [-10000, -10000, 36900, 40.5]}

  const categories4 = ['Increased', 'Diminished', 'Same / No change', 'No violent acts / crime', 'No Response']

  const categories4XY = {'Increased': [0, 450, 42745, 31.2],
                          'Diminished': [450, 850, 30100, 79.4],
                          'Same / No change': [450, 450, 31913, 63.2],
                          'No violent acts / crime':  [0, 850, 36900, 40.5],
                          'No Response': [0, 150, 42200, 48.3],
                          'Yes': [-10000, -10000, 42200, 48.3],
                          'Not': [-10000, -10000, 42745, 31.2],
                          'Not Asked': [-10000, -10000, 36900, 40.5],
                          'We live comfortably on current income': [-10000, -10000, 42745, 31.2],
                          'We survive on current income': [-10000, -10000, 36900, 40.5],
                          'We are struggling with current income': [-10000, -10000, 31913, 63.2],
                          'We are in a very critical situation with current income': [-10000, -10000,  30100, 79.4]}


const margin = {left: 170, top: 50, bottom: 50, right: 20}
const width = 1000 - margin.left - margin.right
const height = 950 - margin.top - margin.bottom

//Read Data, convert numerical categories into floats
//Create the initial visualisation


d3.csv('./data/recent-grads.csv', function(d){
    return {
        Major: d.Major,
        x: randomIntFromInterval(225, 775),
        y: randomIntFromInterval(225, 775),
        Total: d.Number,
        Men: +d.Men,
        Women: +d.Women,
        Median: +d.Median,
        Unemployment: +d.Unemployment_rate,
        Category: d.Major_category,
        ShareWomen: +d.ShareWomen,
        HistCol: +d.Histogram_column,
        Midpoint: +d.midpoint,
        Number: d.number
    };
}).then(data => {
    dataset = data
    console.log(dataset)
    d3.csv('./data/new-recent-grads.csv', function(d){
        return {
            Major: d.Major,
            Total: +d.Number,
            Men: d.Mig_plan,
            Women: d.Violence_level,
            Median: +d.Median,
            Unemployment: +d.Unemployment_rate,
            Category: d.Major_category,
            ShareWomen: +d.ShareWomen,
            HistCol: +d.Histogram_column,
            Midpoint: +d.midpoint,
            Number: d.number
        };
    }).then(data => {
        new_dataset = data
        console.log(new_dataset)
        createScales()
        setTimeout(drawInitial(), 100)
    })



})



const colors = ['#ffcc00', '#ff6666', '#cc0066', '#66cccc', '#f688bb']

//Create all the scales and save to global variables

function createScales(){
    salarySizeScale = d3.scaleLinear(d3.extent(dataset, d => d.Median), [5, 35])
    salaryXScale = d3.scaleLinear(d3.extent(dataset, d => d.Median), [margin.left, margin.left + width+250])
    salaryYScale = d3.scaleLinear([20000, 110000], [margin.top + height, margin.top])
    categoryColorScale = d3.scaleOrdinal(categories, colors)
    shareWomenXScale = d3.scaleLinear(d3.extent(dataset, d => d.ShareWomen), [margin.left, margin.left + width])
    enrollmentScale = d3.scaleLinear(d3.extent(dataset, d => d.Total), [margin.left + 120, margin.left + width - 50])
    enrollmentSizeScale = d3.scaleLinear(d3.extent(dataset, d=> d.Total), [10,60])
    histXScale = d3.scaleLinear(d3.extent(dataset, d => d.Midpoint), [margin.left, margin.left + width])
    histYScale = d3.scaleLinear(d3.extent(dataset, d => d.HistCol), [margin.top + height, margin.top])
}

function createLegend(x, y){
    let svg = d3.select('#legend')

    svg.append('g')
        .attr('class', 'categoryLegend')
        .attr('transform', `translate(${x},${y})`)

    categoryLegend = d3.legendColor()
                            .shape('path', d3.symbol().type(d3.symbolCircle).size(150)())
                            .shapePadding(10)
                            .scale(categoryColorScale)

    d3.select('.categoryLegend')
        .call(categoryLegend)
}

function createSizeLegend(){
    let svg = d3.select('#legend2')
    svg.append('g')
        .attr('class', 'sizeLegend')
        .attr('transform', `translate(100,50)`)

    sizeLegend2 = d3.legendSize()
        .scale(salarySizeScale)
        .shape('circle')
        .shapePadding(15)
        .title('Salary Scale')
        .labelFormat(d3.format("$,.2r"))
        .cells(7)

    d3.select('.sizeLegend')
        .call(sizeLegend2)
}

function createSizeLegend2(){
    let svg = d3.select('#legend3')
    svg.append('g')
        .attr('class', 'sizeLegend2')
        .attr('transform', `translate(50,100)`)

    sizeLegend2 = d3.legendSize()
        .scale(enrollmentSizeScale)
        .shape('circle')
        .shapePadding(55)
        .orient('horizontal')
        .title('Enrolment Scale')
        .labels(['1000', '200000', '400000'])
        .labelOffset(30)
        .cells(3)

    d3.select('.sizeLegend2')
        .call(sizeLegend2)
}

// All the initial elements should be create in the drawInitial function
// As they are required, their attributes can be modified
// They can be shown or hidden using their 'opacity' attribute
// Each element should also have an associated class name for easy reference

function drawInitial(){
    // createSizeLegend()
    // createSizeLegend2()

    let svg = d3.select("#vis")
                    .append('svg')
                    .attr('width', 1000)
                    .attr('height', 950)
                    .attr('opacity', 1)

    // let xAxis = d3.axisBottom(salaryXScale)
    //                 .ticks(4)
    //                 .tickSize(height + 80)

    // let xAxisGroup = svg.append('g')
    //     .attr('class', 'first-axis')
    //     .attr('transform', 'translate(0, 0)')
    //     .call(xAxis)
    //     .call(g => g.select('.domain')
    //         .remove())
    //     .call(g => g.selectAll('.tick line'))
    //         .attr('stroke-opacity', 0.2)
    //         .attr('stroke-dasharray', 2.5)

    // Instantiates the force simulation
    // Has no forces. Actual forces are added and removed as required

    simulation = d3.forceSimulation(dataset)

     // Define each tick of simulation
    simulation.on('tick', () => {
        nodes
            .attr('cx', d => d.x)
            .attr('cy', d => d.y)
    })

    // Stop the simulation until later
    simulation.stop()


    // Selection of all the circles
    nodes = svg
        .selectAll('circle')
        .data(dataset)
        .enter()
        .append('circle')
            .attr('fill', 'black')
            .attr('r', 3)
            .attr('cx', (d, i) => salaryXScale(d.Median) + 5)
            .attr('cy', (d, i) => i * 5.2 + 30)
            .attr('opacity', 0)

    // Add mouseover and mouseout events for all circles
    // Changes opacity and adds border
    svg.selectAll('circle')
        .on('mouseover', mouseOver)
        .on('mouseout', mouseOut)

    function mouseOver(d, i){
        d3.select(this)
            .transition('mouseover').duration(100)
            .attr('opacity', 1)
            .attr('stroke-width', 5)
            .attr('stroke', 'black')

        d3.select('#tooltip')
            .style('left', (d3.event.pageX + 10)+ 'px')
            .style('top', (d3.event.pageY - 25) + 'px')
            .style('display', 'inline-block')
            .html(`<strong>Country:</strong> ${d.Major[0] + d.Major.slice(1,).toLowerCase()}
                <br> <strong>Feelings about Income:</strong> ${d.Category}`)
    }

    function mouseOut(d, i){
        d3.select('#tooltip')
            .style('display', 'none')

        d3.select(this)
            .transition('mouseout').duration(100)
            .attr('opacity', 0.8)
            .attr('stroke-width', 0)
    }

    //Small text label for first graph
    // svg.selectAll('.small-text')
    //     .data(dataset)
    //     .enter()
    //     .append('text')
    //         .text((d, i) => d.Major.toLowerCase())
    //         .attr('class', 'small-text')
    //         .attr('x', margin.left)
    //         .attr('y', (d, i) => i * 5.2 + 30)
    //         .attr('font-size', 7)
    //         .attr('text-anchor', 'end')

    //All the required components for the small multiples charts
    //Initialises the text and rectangles, and sets opacity to 0
    svg.selectAll('.cat-rect')
        .data(categories).enter()
        .append('rect')
            .attr('class', 'cat-rect')
            .attr('x', d => categoriesXY[d][0] + 120 + 1000)
            .attr('y', d => categoriesXY[d][1] + 30)
            .attr('width', 400)
            .attr('height', 30)
            .attr('opacity', 0)
            .attr('fill', 'grey')


    svg.selectAll('.lab-text')
        .data(categories).enter()
        .append('text')
        .attr('class', 'lab-text')
        .attr('opacity', 0)
        .raise()

    svg.selectAll('.lab-text')
        .text(d => d)
        .attr('x', d => categoriesXY[d][0] + 200 + 1000)
        .attr('y', d => categoriesXY[d][1] - 500)
        .attr('font-family', 'Domine')
        .attr('font-size', '12px')
        .attr('font-weight', 700)
        .attr('fill', 'black')
        .attr('text-anchor', 'center')

    svg.selectAll('.lab-text')
            .on('mouseover', function(d, i){
                d3.select(this)
                    .text(d)
            })
            .on('mouseout', function(d, i){
                d3.select(this)
                    .text(d => d)
            })


    // Best fit line for gender scatter plot

    const bestFitLine = [{x: 0, y: 56093}, {x: 1, y: 25423}]
    const lineFunction = d3.line()
                            .x(d => shareWomenXScale(d.x))
                            .y(d => salaryYScale(d.y))

    // Axes for Scatter Plot
    svg.append('path')
        .transition('best-fit-line').duration(430)
            .attr('class', 'best-fit')
            .attr('d', lineFunction(bestFitLine))
            .attr('stroke', 'grey')
            .attr('stroke-dasharray', 6.2)
            .attr('opacity', 0)
            .attr('stroke-width', 3)

    let scatterxAxis = d3.axisBottom(shareWomenXScale)
    let scatteryAxis = d3.axisLeft(salaryYScale).tickSize([width])

    svg.append('g')
        .call(scatterxAxis)
        .attr('class', 'scatter-x')
        .attr('opacity', 0)
        .attr('transform', `translate(0, ${height + margin.top})`)
        .call(g => g.select('.domain')
            .remove())

    svg.append('g')
        .call(scatteryAxis)
        .attr('class', 'scatter-y')
        .attr('opacity', 0)
        .attr('transform', `translate(${margin.left - 20 + width}, 0)`)
        .call(g => g.select('.domain')
            .remove())
        .call(g => g.selectAll('.tick line'))
            .attr('stroke-opacity', 0.2)
            .attr('stroke-dasharray', 2.5)

    // Axes for Histogram

    let histxAxis = d3.axisBottom(enrollmentScale)

    svg.append('g')
        .attr('class', 'enrolment-axis')
        .attr('transform', 'translate(0, 700)')
        .attr('opacity', 0)
        .call(histxAxis)
}

//Cleaning Function
//Will hide all the elements which are not necessary for a given chart type

function clean(chartType){
    let svg = d3.select('#vis').select('svg')
    if (chartType !== "isScatter") {
        svg.select('.scatter-x').transition().attr('opacity', 0)
        svg.select('.scatter-y').transition().attr('opacity', 0)
        svg.select('.best-fit').transition().duration(200).attr('opacity', 0)
    }
    if (chartType !== "isMultiples"){
        svg.selectAll('.lab-text').transition().attr('opacity', 0)
            .attr('x', 1800)
        svg.selectAll('.cat-rect').transition().attr('opacity', 0)
            .attr('x', 1800)
    }
    if (chartType !== "isFirst" || chartType === "isFirst"){
        svg.select('.first-axis').transition().attr('opacity', 0)
        svg.selectAll('.small-text').transition().attr('opacity', 0)
            .attr('x', -200)
        svg.selectAll('circle').transition().attr('opacity', 1)
    }
    if (chartType !== "isHist"){
        svg.selectAll('.hist-axis').transition().attr('opacity', 0)
    }
    if (chartType !== "isBubble"){
        svg.select('.enrolment-axis').transition().attr('opacity', 0)
    }
}

//First draw function



function draw1(){
    //Stop simulation
    simulation.stop()

    let svg = d3.select("#vis")
                    .select('svg')
                    .attr('width', 1000)
                    .attr('height', 950)

    clean('none')

    d3.select('.categoryLegend').transition().remove()

    // svg.select('.first-axis')
    //     .attr('opacity', 1)

    // svg.selectAll('circle')
    //     .transition().duration(500).delay(100)
    //     .attr('fill', 'black')
    //     .attr('r', 3)
    //     .attr('cx', (d, i) => salaryXScale(d.Median)+5)
    //     .attr('cy', (d, i) => i * 5.2 + 30)

    svg.selectAll('circle')
        .transition()
        .attr('r', d => 15)
        .attr('fill', d => '#ff6666')

    simulation
        .force('forceX', d3.forceX(500))
        .force('forceY', d3.forceY(500))
        .force('collide', d3.forceCollide(d => 10 * 1.6 + 4))
        .alpha(0.6).alphaDecay(0.05).restart()
}


function draw2(){
    let svg = d3.select("#vis").select('svg')

    clean('none')

    svg.selectAll('circle')
        .transition().duration(300).delay((d, i) => 2)
        .attr('r', d => 3)
        .attr('fill', d => categoryColorScale(d.Category))
        .attr('cx', (d, i) => categoriesXY[d.Category][0] + 200)
        .attr('cy', (d, i) => categoriesXY[d.Category][1] - 50)

    simulation
        .force('charge', d3.forceManyBody().strength(0.03))
        .force('forceX', d3.forceX(d => categoriesXY[d.Category][0] + 200))
        .force('forceY', d3.forceY(d => categoriesXY[d.Category][1] - 50))
        .force('collide', d3.forceCollide(d => 5))
        .alphaDecay([0.2])

    // //Reheat simulation and restart
    simulation.alpha(0.9).restart()

    createLegend(20, 50)
}

function incomeFeelBubbles(){
    let svg = d3.select("#vis").select('svg')
    clean('isMultiples')

    simulation = d3.forceSimulation(dataset)

     // Define each tick of simulation
    simulation.on('tick', () => {
        nodes
            .attr('cx', d => d.x)
            .attr('cy', d => d.y)
    })

    // Stop the simulation until later
    simulation.stop()

    nodes = svg.selectAll('circle')
        .data(dataset)

    svg.selectAll('circle')
        .transition().duration(400).delay((d, i) => 2)
        .attr('r', d => 2)
        .attr('fill', d => categoryColorScale(d.Category))
        .attr('cx', (d, i) => categoriesXY[d.Category][0] + 500)
        .attr('cy', (d, i) => categoriesXY[d.Category][1] - 50)

    svg.selectAll('.cat-rect')
        .data(categories)

    svg.selectAll('.cat-rect').transition().duration(300).delay((d, i) => i * 30)
        .attr('opacity', 0.2)
        .attr('x', d => categoriesXY[d][0] + 120)
        .attr('y', d => categoriesXY[d][1] + 30)
        .attr('width', 400)
        .attr('height', 30)

    svg.selectAll('.lab-text')
        .data(categories)

    svg.selectAll('.lab-text').transition().duration(300).delay((d, i) => i * 30)
        .text(d => d)
        .attr('x', d => categoriesXY[d][0] + 275 - 2*d.length)
        .attr('y', d => categoriesXY[d][1] + 50)
        .attr('opacity', 1)

    simulation
        .force('charge', d3.forceManyBody().strength([0.5]))
        .force('forceX', d3.forceX(d => categoriesXY[d.Category][0] + 350)) // d => categoriesXY[d.Category][0] + 350) sets location of where bubbles appear
        .force('forceY', d3.forceY(d => categoriesXY[d.Category][1] - 100)) 
        .force('collide', d3.forceCollide(d => 6.5))
        .alpha(0.7).alphaDecay(0.02).restart()

}

function violenceBubbles() {
  let svg = d3.select("#vis").select('svg')
  clean('isMultiples2')

  simulation = d3.forceSimulation(new_dataset)

   // Define each tick of simulation
  simulation.on('tick', () => {
      nodes
          .attr('cx', d => d.x)
          .attr('cy', d => d.y)
  })

  // Stop the simulation until later
  simulation.stop()

  nodes = svg.selectAll('circle')
      .data(new_dataset)

  svg.selectAll('circle')
      .transition().duration(400).delay((d, i) => 2)
      .attr('r', d => 4)
      .attr('fill', d => categoryColorScale(d.Women))
      .attr('cx', (d, i) => 500)
      .attr('cy', (d, i) => 500)
      .attr('opacity', 1)

  let temp_dict = {'Increased': 0,
                  'Diminished': 100,
                  'Same / No change': 200,
                  'No violent acts / crime':  300,
                  'No Response': 400}

  svg.selectAll('.cat-rect')
      .data(categories4)

  svg.selectAll('.cat-rect').transition().duration(300).delay((d, i) => i * 30)
      .attr('opacity', 0.2)
      .attr('x', d => categories4XY[d][0] + 170)
      .attr('y', d => categories4XY[d][1] + 30)
      .attr('width', 200)
      .attr('height', 30)
  svg.selectAll('.lab-text')
      .data(categories4)

  svg.selectAll('.lab-text').transition().duration(300).delay((d, i) => i * 30)
      .text(d => d)
      .attr('x', d => categories4XY[d][0] + 70 + 200 - d.length*2.5)
      .attr('y', d => categories4XY[d][1] + 50)
      .attr('opacity', 1)


  simulation
      .force('charge', d3.forceManyBody().strength([0.5]))
      .force('forceX', d3.forceX(d => categories4XY[d.Women][0] + 285))
      .force('forceY', d3.forceY(d => categories4XY[d.Women][1] - 100))
      .force('collide', d3.forceCollide(d => 6.5))
      .alpha(0.7).alphaDecay(0.02).restart()
}

function draw35() {
  let svg = d3.select("#vis").select('svg')
  clean('isMultiples2')

  svg.selectAll('circle')
      .transition().duration(400).delay((d, i) => 2)
      .attr('r', d => 4)
      .attr('fill', d => categoryColorScale(d.Category))
      .attr('cx', (d, i) => 500)
      .attr('cy', (d, i) => 500)
      .attr('opacity', 1)



  svg.selectAll('.cat-rect')
      .data(categories2)

  svg.selectAll('.cat-rect').transition().duration(300).delay((d, i) => i * 30)
      .attr('opacity', 0.2)
      .attr('x', d => categories2XY[d][0] + 70)
      .attr('y', d => categories2XY[d][1] + 30)
      .attr('width', 200)
      .attr('height', 30)
  svg.selectAll('.lab-text')
      .data(categories2)

  svg.selectAll('.lab-text').transition().duration(300).delay((d, i) => i * 30)
      .text(d => d)
      .attr('x', d => categories2XY[d][0] + 70 + 100 - d.length*2)
      .attr('y', d => categories2XY[d][1] + 50)
      .attr('opacity', 1)

  simulation
      .force('charge', d3.forceManyBody().strength([0.5]))
      .force('forceX', d3.forceX(d => categories2XY[d.Category][0] + 185))
      .force('forceY', d3.forceY(d => categories2XY[d.Category][1] - 100))
      .force('collide', d3.forceCollide(d => 6.5))
      .alpha(0.7).alphaDecay(0.02).restart()
}

function draw36() {
  let svg = d3.select("#vis").select('svg')
  clean('isMultiples2')

  svg.selectAll('circle')
      .transition().duration(400).delay((d, i) => 2)
      .attr('r', d => 4)
      .attr('fill', d => categoryColorScale(d.Men))
      .attr('cx', (d, i) => 500)
      .attr('cy', (d, i) => 500)
      .attr('opacity', 1)

  svg.selectAll('.cat-rect')
      .data(categories3)

  svg.selectAll('.cat-rect').transition().duration(300).delay((d, i) => i * 30)
      .attr('opacity', 0.2)
      .attr('x', d => categories3XY[d][0] + 70)
      .attr('y', d => categories3XY[d][1] + 30)
      .attr('width', 200)
      .attr('height', 30)

  svg.selectAll('.lab-text')
      .data(categories3)

  svg.selectAll('.lab-text').transition().duration(300).delay((d, i) => i * 30)
      .text(d => d)
      .attr('x', d => categories3XY[d][0] + 70 + 100 - d.length*2)
      .attr('y', d => categories3XY[d][1] + 50)
      .attr('opacity', 1)

  simulation
      .force('charge', d3.forceManyBody().strength([0.5]))
      .force('forceX', d3.forceX(d => categories3XY[d.Men][0] + 185))
      .force('forceY', d3.forceY(d => categories3XY[d.Men][1] - 100))
      .force('collide', d3.forceCollide(d => 6.5))
      .alpha(0.7).alphaDecay(0.02).restart()
}

function draw8(){
    clean('none')

    let svg = d3.select('#vis').select('svg')
    // svg
    //     .selectAll('circle')
    //     .data(dataset)
    //     .enter()
    //     .append('circle')
    //         .attr('fill', 'black')
    //         .attr('r', 3)
    //         .attr('cx', (d, i) => salaryXScale(d.Median) + 5)
    //         .attr('cy', (d, i) => i * 5.2 + 30)
    //         .attr('opacity', 0)

    svg.selectAll('circle')
        .transition().duration(100).delay((d, i) => 2)
        .attr('r', d => 2)
        .attr('cx', (d, i) => d.x )
        .attr('cy', (d, i) => d.y )
        .attr('fill', d => '#ff6666')
        .attr('opacity',1)

    simulation
        .force('forceX', d3.forceX(d => d.x))
        .force('forceY', d3.forceY(d => d.y))
        .force('collide', d3.forceCollide(d => 6.5))
        .alpha(0.7).alphaDecay(0.5).restart()

}

function randomIntFromInterval(min, max) { // min and max included
  return Math.floor(Math.random() * (max - min + 1) + min)
}

//
// svg.selectAll('circle')
//     .transition().duration(400).delay((d, i) => 2)
//     .attr('r', d => 2)
//     .attr('fill', d => categoryColorScale(d.Category))
//     .attr('cx', (d, i) => categoriesXY[d.Category][0] + 500)
//     .attr('cy', (d, i) => categoriesXY[d.Category][1] - 50)
//
// svg.selectAll('.cat-rect').transition().duration(300).delay((d, i) => i * 30)
//     .attr('opacity', 0.2)
//     .attr('x', d => categoriesXY[d][0] + 120)
//
// svg.selectAll('.lab-text').transition().duration(300).delay((d, i) => i * 30)
//     .text(d => d)
//     .attr('x', d => categoriesXY[d][0] + 275 - 2*d.length)
//     .attr('y', d => categoriesXY[d][1] + 50)
//     .attr('opacity', 1)
//
// simulation
//     .force('charge', d3.forceManyBody().strength([0.5]))
//     .force('forceX', d3.forceX(d => categoriesXY[d.Category][0] + 350))
//     .force('forceY', d3.forceY(d => categoriesXY[d.Category][1] - 100))
//     .force('collide', d3.forceCollide(d => 6.5))
//     .alpha(0.7).alphaDecay(0.02).restart()

function allBubbles() {
  clean('none')
  let svg = d3.select('#vis').select('svg')
  svg.selectAll('circle')
      .data(dataset)
      .enter()
      .append('circle')
          .attr('fill', 'black')
          .attr('r', 3)
          .attr('cx', (d, i) => salaryXScale(d.Median) + 5)
          .attr('cy', (d, i) => i * 5.2 + 30)
          .attr('opacity', 0)
}

//Array of all the graph functions
//Will be called from the scroller functionality

let activationFunctions = [
    allBubbles,
    draw8,
    incomeFeelBubbles,
    violenceBubbles,
    draw35,
    draw36
]

// let activationFunctions = [
//   draw7,
//   draw8
// ]
//All the scrolling function
//Will draw a new graph based on the index provided by the scroll


let scroll = scroller()
    .container(d3.select('#graphic'))
scroll()

let lastIndex, activeIndex = 0

scroll.on('active', function(index){
    d3.selectAll('.step')
        .transition().duration(500)


    activeIndex = index
    let sign = (activeIndex - lastIndex) < 0 ? -1 : 1;
    let scrolledSections = d3.range(lastIndex + sign, activeIndex + sign, sign);
    scrolledSections.forEach(i => {
        activationFunctions[i]();
    })
    lastIndex = activeIndex;

})

scroll.on('progress', function(index, progress){
    if (index == 2 & progress > 0.7){

    }
})

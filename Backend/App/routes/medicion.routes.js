module.exports = app => {
    const mediciones = require("../controllers/medicion.controllers.js");

    var router = require("express").Router();

    //Crear una nueva medicion
    router.post("/", mediciones.create);

    //Encontrar todas las mediciones
    router.get("/", mediciones.findAll);

    //Encontrar todo parametro mediciones
    router.get("/Parametro", mediciones.findAllParametro);

    //Encontrar una sola medicion con el id
    router.get("/:id", mediciones.findOne);

    //Actualizar una medicion con el id
    router.put("/:id", mediciones.update);

    //Borrar una medicion con el id
    router.delete("/:id", mediciones.delete);

    //Borrar todas las mediciones
    router.delete("/", mediciones.deleteAll);

    app.use('/api/mediciones', router);
};
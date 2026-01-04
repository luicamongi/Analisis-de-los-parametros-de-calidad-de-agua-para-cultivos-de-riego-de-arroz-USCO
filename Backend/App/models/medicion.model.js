module.exports = (sequelize, Sequelize) => {
    const Medicion = sequelize.define("medicion", {
        Parametro: {
            type: Sequelize.STRING
        },
        Valor: {
            type: Sequelize.FLOAT
        },
        Ubicacion: {
            type: Sequelize.STRING
        }
    });

    return Medicion;
};

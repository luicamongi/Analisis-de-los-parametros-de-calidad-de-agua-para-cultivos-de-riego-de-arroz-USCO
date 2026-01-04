module.exports = {
    HOST: "localhost",
    USER: "postgres",
    PASSWORD: "0513",
    DB: "mediciones_calidad",
    dialect: "postgres",
    pool: {
        max: 5,
        min: 0,
        acquire: 3000000,
        idle:10000
    }
};
